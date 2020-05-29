import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 6 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
        return logits, bbox_reg, centerness


class FCOSFuseFPN(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(FCOSFuseFPN, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        tran_clstower = []
        tran_regretower = []
        tran_centertower = []
        for i in range(1,6):
            tran_clstower.append(
                nn.ConvTranspose2d(i,i,2,2)
            )
            tran_centertower.append(
                nn.ConvTranspose2d(i,i,2,2)
            )
            tran_regretower.append(
                nn.ConvTranspose2d(6*i,6*i,2,2)
            )

        for modules in [tran_clstower, tran_centertower, tran_regretower]:
            for l in modules:
                if isinstance(l, nn.ConvTranspose2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        self.tran_clstower = nn.ModuleList(tran_clstower)
        self.tran_regretower = nn.ModuleList(tran_regretower)
        self.tran_centertower = nn.ModuleList(tran_centertower)

        self.fuse_cls = nn.Conv2d(num_classes*5, num_classes, 3, 1, 1)
        self.fuse_reg = nn.Conv2d(dense_points*6*5, dense_points*6, 3, 1, 1)
        self.fuse_cet = nn.Conv2d(dense_points*5, dense_points, 3, 1, 1)


    def forward(self, features, targets=None):
        box_cls, box_regression, centerness = features
        box_cls_tmp = box_cls[-1]
        box_regression_tmp = box_regression[-1]
        centerness_tmp = centerness[-1]
        for i in range(4):
            box_cls_tmp = torch.cat([self.tran_clstower[i](box_cls_tmp),box_cls[-i-2]], dim=1)
            box_regression_tmp = torch.cat([self.tran_regretower[i](box_regression_tmp), box_regression[-i-2]], dim=1)
            centerness_tmp = torch.cat([self.tran_centertower[i](centerness_tmp), centerness[-i-2]], dim=1)

        box_cls = self.tran_clstower[-1](box_cls_tmp)
        box_regression = self.tran_regretower[-1](box_regression_tmp)
        centerness = self.tran_centertower[-1](centerness_tmp)

        box_cls = self.fuse_cls(box_cls)
        box_regression = self.fuse_reg(box_regression)
        centerness = self.fuse_cet(centerness)

        return box_cls, box_regression, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)
        fuseFPN = FCOSFuseFPN(cfg, in_channels)
        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.fuseFPN = fuseFPN
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        box_cls_f, box_regression_f, centerness_f = self.fuseFPN((box_cls, box_regression, centerness))
        locations = self.compute_locations(features)
        locations_f = self.compute_locations_f(box_cls_f.shape[-2:], box_cls_f.device)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations_f(self, image_sizes, device):
        locations = []
        h, w = image_sizes
        locations_per_level = self.compute_locations_per_level(
            h, w, 4,
            device
        )
        locations.append(locations_per_level)
        return locations

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        locations = self.get_dense_locations(locations, stride, device)
        return locations

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device)
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            print("dense points only support 1, 4, 5")
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
