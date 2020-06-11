"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import numpy as np


INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.loc_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets_f(self, locations, targets, h_f, w_f):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        hm = np.zeros((len(targets),1,h_f,w_f), dtype=np.float32)
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            cx = (bboxes[:,0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:,1] + bboxes[:, 3]) * 0.5
            c_w = bboxes[:,2] - bboxes[:,0]
            c_h = bboxes[:,3] - bboxes[:,1]
            
            center_x = xs[:, None] - cx
            center_y = ys[:, None] - cy
            is_in_boxes = (center_x<=2) & (center_y<=2) & (center_x>-2) & (center_y>-2)

            if is_in_boxes.nonzero().shape[0] != cx.shape[0]:
                import pdb; pdb.set_trace()
                raise ValueError("In FPNfuse must one point match target!")

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            # support class num is n0
            cord_x = (is_in_boxes.nonzero()[:,0] % w_f).cpu().numpy()
            cord_y = (is_in_boxes.nonzero()[:,0] / w_f).cpu().numpy()
            cord_w = c_w[is_in_boxes.nonzero()[:,1]].cpu().numpy()
            cord_h = c_h[is_in_boxes.nonzero()[:,1]].cpu().numpy()
            cord_c = labels_per_im[is_in_boxes.nonzero()[:,0]]
            
            for i in range(1):
                for n_cord in range(cord_x.shape[0]):
                    if cord_c[n_cord] == i+1:
                        hm[im_i, i,:,:] = self.draw_umich_gaussian(hm[im_i, i,:,:], (cord_x[n_cord], cord_y[n_cord]), (cord_w[n_cord], cord_h[n_cord]))
            '''
            im_hm = hm[im_i,i,:,:][...,None]
            import cv2
            cv2.imshow("res",im_hm)
            cv2.waitKey()
            exit(0)
            import pdb; pdb.set_trace()
            '''
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        labels_level_first = torch.cat([labels_per_im for labels_per_im in labels], dim=0)
        reg_targets_level_first = torch.cat([reg_targets_per_im for reg_targets_per_im in reg_targets], dim=0)
        hm = torch.from_numpy(hm).to(labels_level_first.device)
        return labels_level_first, reg_targets_level_first, hm

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            '''
            repeatX = l.shape[0]
            repeatY = t.shape[0]
            
            scales = torch.where(object_sizes_of_interest[:,1]==INF, \
                object_sizes_of_interest.new_tensor(800).expand(object_sizes_of_interest.shape[0]), \
                    object_sizes_of_interest[:,1])
            scales *= 1.5
            vw = (bboxes[:, 4][None].repeat(repeatX, 1) / scales[:,None]).unsqueeze(2)
            vh = (bboxes[:, 5][None].repeat(repeatY, 1) / scales[:,None]).unsqueeze(2)
            '''
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets, locations_f=None, res_f=None):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1) // self.dense_points
        h_f, w_f = res_f[0].shape[-2:]
        
        labels, reg_targets = self.prepare_targets(locations, targets)
        labels_f, reg_targets_f, hm  = self.prepare_targets_f(locations_f, targets, h_f, w_f)
        box_cls_f, box_regression_f = res_f

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []

        hm = hm.permute(0,2,3,1).reshape(-1, num_classes)
        box_cls_f = box_cls_f.permute(0,2,3,1).reshape(-1, num_classes)
        box_regression_f = box_regression_f.permute(0,2,3,1).reshape(-1, 4)
        labels_f = labels_f.reshape(-1)
        reg_targets_f = reg_targets_f.reshape(-1, 4)

        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].permute(0, 2, 3, 1).reshape(-1))
        
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        pos_inds_f = torch.nonzero(labels_f > 0).squeeze(1)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        box_regression_f = box_regression_f[pos_inds_f]
        reg_targets_f = reg_targets_f[pos_inds_f]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets,
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
        
        reg_loss_f = self.box_reg_loss_func(
            box_regression_f, reg_targets_f
        )
        
        cls_loss_f = self.cls_loss_func(
            box_cls_f,
            labels_f.int(), hm
        ) / (pos_inds_f.numel() + N)
        
        #import pdb; pdb.set_trace()
        #print("reg_loss:{}".format(reg_loss))
        #print("reg_loss_f:{}".format(reg_loss_f))
        #import pdb; pdb.set_trace()

        cls_loss_f = cls_loss_f
        reg_loss_f = reg_loss_f*0.1
        return cls_loss, reg_loss, centerness_loss, reg_loss_f, cls_loss_f

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)


    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, det_size, min_overlap=0.7, k=1):
        radius = int(self.gaussian_radius(det_size, min_overlap))
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]
            
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
