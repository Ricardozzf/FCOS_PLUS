import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None, a=1):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        iou_wh = 1
        giou_wh = 1
        if pred.shape[1] !=4:
            pred_w, pred_h = pred[:,4], pred[:,5]
            target_w, target_h = target[:,4], target[:,5]
            insert_wh = torch.min(pred_w, target_w) * torch.min(pred_h, target_h)
            bb_wh = torch.max(pred_w, target_w) * torch.max(pred_h, target_h)
            union_wh = target_w * target_h + pred_w * pred_h - insert_wh
            iou_wh = insert_wh / union_wh
            giou_wh = iou_wh - (bb_wh - union_wh) / bb_wh
        
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious) + -torch.log(iou_wh) * a
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious + a*(1 - iou_wh)
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious + a*(1 - giou_wh)
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
