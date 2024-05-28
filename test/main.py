import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import PredictModel, Predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Test Object Detection3D')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='./data/input', type=str)
    parser.add_argument('--output_path', default='./data/output', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/Resnet3D/100.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_detection = PredictModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_detection = Predictor(
        device=device,
        model=model_detection,
    )
    os.makedirs(output_path, exist_ok=True)
    dcm_dir = os.path.join(input_path, "dcms")

    for sample in tqdm(os.listdir(dcm_dir)):  
        sitk_img = load_scans(os.path.join(dcm_dir, sample))
        volume = sitk.GetArrayFromImage(sitk_img)
        score, prop_bbox = predictor_detection.predict(volume)
        box_mask = np.zeros_like(volume).astype("uint8")
        z_min, y_min, x_min, z_max, y_max, x_max = prop_bbox
        box_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
        box_itk = sitk.GetImageFromArray(box_mask)
        box_itk.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img, os.path.join(output_path, f'{sample}.dcm.nii.gz'))
        sitk.WriteImage(box_itk, os.path.join(output_path, f'{sample}-confidence{score:.2f}.box.nii.gz'))
        
        data_for_metrics = os.path.join(output_path, "data_for_metrics")
        os.makedirs(data_for_metrics, exist_ok=True)
        label_path = os.path.join(input_path, "labels", f'{sample}.box.nii.gz')
        if os.path.exists(label_path):
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype("uint8")
            np.savez_compressed(os.path.join(data_for_metrics, f'{sample}.npz'), pred=box_mask, label=label)



if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )