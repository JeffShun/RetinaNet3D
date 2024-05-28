import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, img_size, box_scale_factor):
        super(Decoder, self).__init__()
        self.img_size = img_size 
        self.box_scale_factor = box_scale_factor

    def forward(self, cls_heads, reg_heads, batch_anchors):
        # cls_heads: [batch, nbox]
        # reg_heads: [batch, nbox, 6]
        # batch_anchors: [batch, nbox, 6]
        with torch.no_grad():
            # 选择分数最高的框
            batchsize = cls_heads.shape[0]
            max_scores, max_indices = torch.max(cls_heads, dim=1)  
            max_indices = max_indices.long()
            max_reg_heads = reg_heads[torch.arange(batchsize), max_indices]           # shape: [batch, 6]
            max_batch_anchors = batch_anchors[torch.arange(batchsize), max_indices]   # shape: [batch, 6]

            # 将回归头的预测值转换为边界框坐标
            image_pred_bboxes = self.tz_ty_tx_td_th_tw_to_z1_y1_x1_z2_y2_x2_bboxes(max_reg_heads, max_batch_anchors)
            return max_scores, image_pred_bboxes
        

    def tz_ty_tx_td_th_tw_to_z1_y1_x1_z2_y2_x2_bboxes(self, reg_heads, anchors):
        """
        将回归头的预测值转换为边界框坐标
        """
        anchors_dhw = anchors[:, 3:] - anchors[:, :3]
        anchors_ctr = anchors[:, :3] + 0.5 * anchors_dhw

        if self.box_scale_factor:
            factor = torch.tensor([self.box_scale_factor], device=anchors.device)
            reg_heads = reg_heads * factor

        pred_bboxes_dhw = torch.exp(reg_heads[:, 3:]) * anchors_dhw
        pred_bboxes_ctr = reg_heads[:, :3] * anchors_dhw + anchors_ctr

        pred_bboxes_z_min_y_min_x_min = pred_bboxes_ctr - 0.5 * pred_bboxes_dhw
        pred_bboxes_z_max_y_max_x_max = pred_bboxes_ctr + 0.5 * pred_bboxes_dhw

        pred_bboxes = torch.cat([pred_bboxes_z_min_y_min_x_min, pred_bboxes_z_max_y_max_x_max], dim=1)

        # 将边界框坐标限制在图像范围内
        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], min=0)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=self.img_size[0] - 1)
        pred_bboxes[:, 4] = torch.clamp(pred_bboxes[:, 4], max=self.img_size[1] - 1)
        pred_bboxes[:, 5] = torch.clamp(pred_bboxes[:, 5], max=self.img_size[2] - 1)
        return pred_bboxes
