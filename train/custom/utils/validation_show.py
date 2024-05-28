import numpy as np
import SimpleITK as sitk
import os


def cal_IOU(pred, label):
    intersection = (pred*label).sum()
    union = pred.sum() + label.sum() - intersection
    iou = intersection / union
    return iou

def save_validation_result(images, labels, scores, pred_boxes, save_dir):
    for i in range(images.shape[0]):
        img = (images[i, 0].cpu().numpy()*255).astype("uint8")
        label = labels[i, 0].cpu().numpy().astype("int32")
        score = scores[i].cpu().numpy()
        pred_box = pred_boxes[i].cpu().numpy().astype("int32")

        box_mask = np.zeros_like(img).astype("uint8")
        z_min, y_min, x_min, z_max, y_max, x_max = pred_box
        box_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1

        label_mask = np.zeros_like(img).astype("uint8")
        z_min, y_min, x_min, z_max, y_max, x_max = label
        label_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1

        box_itk = sitk.GetImageFromArray(box_mask)
        label_itk = sitk.GetImageFromArray(label_mask)
        img_itk = sitk.GetImageFromArray(img)

        iou = cal_IOU(box_mask, label_mask)

        sitk.WriteImage(box_itk, os.path.join(save_dir, f'{i+1}-confidence{score:.2f}-iou{iou:.2f}.box.nii.gz'))
        sitk.WriteImage(label_itk, os.path.join(save_dir, f'{i+1}.label.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(save_dir, f'{i+1}.dcm.nii.gz'))
