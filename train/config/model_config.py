import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResNet3D
from custom.model.neck import PyramidFeatures
from custom.model.anchor import Anchors
from custom.model.loss import Focal_Loss
from custom.model.head import Detection_Head
from custom.model.decoder import Decoder
from custom.model.network import Detection_Network

class network_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # img
    img_size = (64, 256, 256)
    
    # network
    in_channel = 1
    base_channel = 32
    pyramid_levels = [3, 4, 5]
    fpn_sizes = [base_channel*2, base_channel*4, base_channel*8]
    p_feature_size = 256
    num_anchors = 3
    num_classes = 1
    h_feature_size = 256
    box_scale_factor = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    network = Detection_Network(
        backbone = ResNet3D(
            in_channel=in_channel, 
            base_channel=base_channel,
            layers=[3, 4, 6, 3]
            ),
        neck = PyramidFeatures(
            fpn_sizes = fpn_sizes,
            feature_size=p_feature_size
        ),
        anchor_generator=Anchors(
            anchor_path=work_dir + "/train_data/processed_data/anchors.txt",
            pyramid_levels=pyramid_levels
        ),
        head=Detection_Head(
            num_features_in=p_feature_size,
            num_classes=num_classes,
            num_anchors=num_anchors,
            feature_size=h_feature_size
        ),
        decoder=Decoder(
            img_size=img_size,
            box_scale_factor=box_scale_factor,
        ),
    apply_sync_batchnorm=False
    )

    # loss function
    train_loss_f = Focal_Loss(alpha=0.75, gamma=2.0, pos_thresh=0.4, neg_thresh=0.2, scale_factor=box_scale_factor)
    valid_loss_f = Focal_Loss(alpha=0.75, gamma=2.0, pos_thresh=0.4, neg_thresh=0.2, scale_factor=box_scale_factor)

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size),
            random_gamma_transform(gamma_range=[0.8, 1.2], prob=0.5),
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size)
            ])
        )
    
    # train dataloader
    batchsize = 3
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [40,80,120]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 150
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/Resnet3D"
    checkpoints_dir = work_dir + '/checkpoints/Resnet3D'
    load_from = work_dir + '/checkpoints/Resnet3D/none.pth'
