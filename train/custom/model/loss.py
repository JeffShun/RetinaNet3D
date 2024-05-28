import torch
import torch.nn as nn

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_thresh=0.5, neg_thresh=0.4, scale_factor=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.scale_factor = scale_factor
        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, classifications, regressions, anchors, labels):
        device = regressions.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_depths = anchor[:, 3] - anchor[:, 0]
        anchor_heights = anchor[:, 4] - anchor[:, 1]
        anchor_widths = anchor[:, 5] - anchor[:, 2]
        anchor_ctr_z = anchor[:, 0] + 0.5 * anchor_depths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        anchor_ctr_x = anchor[:, 2] + 0.5 * anchor_widths

        for j in range(batch_size):

            classification = classifications[j, :]
            regression = regressions[j, :, :]

            bbox_annotation = labels[j, :, :]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = self.calc_iou(anchor, bbox_annotation)  # num_anchors

            # compute the loss for classification
            targets = torch.ones(classification.shape, device=device) * -1
            negative_indices = torch.lt(IoU, self.neg_thresh)
            targets[negative_indices] = 0

            positive_indices = torch.ge(IoU, self.pos_thresh)
            targets[positive_indices] = 1

            alpha_factor = torch.ones(targets.shape, device=device) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            # bce = self.bce_loss(classification, targets)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=device))
            classification_losses.append(cls_loss.sum() / torch.clamp(torch.ne(targets, -1.0).sum(), min=1.0))
            # compute the loss for regression
            if positive_indices.sum() > 0:

                anchor_depths_pi = anchor_depths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_widths_pi = anchor_widths[positive_indices]

                anchor_ctr_z_pi = anchor_ctr_z[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]

                gt_depths = bbox_annotation[:, 3] - bbox_annotation[:, 0]
                gt_heights = bbox_annotation[:, 4] - bbox_annotation[:, 1]
                gt_widths = bbox_annotation[:, 5] - bbox_annotation[:, 2]

                gt_ctr_z = bbox_annotation[:, 0] + 0.5 * gt_depths
                gt_ctr_y = bbox_annotation[:, 1] + 0.5 * gt_heights
                gt_ctr_x = bbox_annotation[:, 2] + 0.5 * gt_widths
 
                # 限制框的长宽高至少为1
                gt_depths = torch.clamp(gt_depths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                gt_widths = torch.clamp(gt_widths, min=1)

                # 计算每个anchor对应的预测目标：平移和缩放参数
                targets_dz = (gt_ctr_z - anchor_ctr_z_pi) / anchor_depths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi

                targets_sz = torch.log(gt_depths / anchor_depths_pi)
                targets_sy = torch.log(gt_heights / anchor_heights_pi)
                targets_sx = torch.log(gt_widths / anchor_widths_pi)

                # 将预测目标进行标准化，使用了预设的缩放因子
                targets = torch.stack((targets_dz, targets_dy, targets_dx, targets_sz, targets_sy, targets_sx))
                targets = targets.t()
                if self.scale_factor:
                    targets = targets / torch.tensor([self.scale_factor], device=device)

                # smooth L1损失函数
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0, device=device).float())

        return {"c_loss":torch.stack(classification_losses).mean(), "r_loss":torch.stack(regression_losses).mean()} 


    def calc_iou(self, a, b):
        area = (b[:, 3] - b[:, 0]) * (b[:, 4] - b[:, 1]) * (b[:, 5] - b[:, 2])
        
        inter_d = torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 0], b[:, 0])
        inter_w = torch.min(a[:, 4], b[:, 4]) - torch.max(a[:, 1], b[:, 1])
        inter_h = torch.min(a[:, 5], b[:, 5]) - torch.max(a[:, 2], b[:, 2])

        inter_d = torch.clamp(inter_d, min=0)
        inter_w = torch.clamp(inter_w, min=0)
        inter_h = torch.clamp(inter_h, min=0)

        ua = (a[:, 3] - a[:, 0]) * (a[:, 4] - a[:, 1]) * (a[:, 5] - a[:, 2]) + area - inter_d * inter_w * inter_h
        ua = torch.clamp(ua, min=1e-8)
        intersection = inter_d * inter_w * inter_h
        IoU = intersection / ua

        return IoU
    