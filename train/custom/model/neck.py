# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class PyramidFeatures(nn.Module):
    def __init__(self, fpn_sizes, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv3d(fpn_sizes[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv3d(fpn_sizes[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv3d(fpn_sizes[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

if __name__ == "__main__":
    c3, c4, c5 = torch.rand(1,64,8,32,32).cuda(), torch.rand(1,128,4,16,16).cuda(), torch.rand(1,256,2,8,8).cuda()
    neck = PyramidFeatures(
        fpn_sizes=[64, 128, 256],
        feature_size=256
        ).cuda()
    P3_x, P4_x, P5_x = neck([c3, c4, c5])
    print(P3_x.shape, P4_x.shape, P5_x.shape)