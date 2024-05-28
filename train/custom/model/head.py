import torch
import torch.nn as nn


class Detection_Head(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_classes: int,
        num_anchors: int,
        feature_size: int,
    ):
        super(Detection_Head, self).__init__()
        self.num_classes = num_classes
        self.regression_head = RegressionModel(num_features_in, num_anchors, feature_size)
        self.classification_head = ClassificationModel(num_features_in, num_anchors, num_classes, feature_size)

    def forward(self, features):
        regressions = torch.cat([self.regression_head(feature) for feature in features], dim=1)
        classifications = torch.cat([self.classification_head(feature) for feature in features], dim=1)
        return regressions, classifications


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv3d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv3d(feature_size, num_anchors * 6, kernel_size=3, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x z x y x x, with C = 6*num_anchors
        out = out.permute(0, 2, 3, 4, 1)

        return out.contiguous().view(out.shape[0], -1, 6)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=1, feature_size=256):
        super(ClassificationModel, self).__init__()
        assert num_classes == 1
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv3d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv3d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, input):
        out = self.conv1(input)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x z x y x x, with C = n_classes x n_anchors
        out1 = out.permute(0, 2, 3, 4, 1)
        batch_size, z, y, x, channels = out1.shape
        out2 = out1.view(batch_size, z, y, x, self.num_anchors, self.num_classes)

        return out2.contiguous().view(input.shape[0], -1)


if __name__ == "__main__":
    features = [torch.rand(1,256,8,32,32).cuda(), torch.rand(1,256,4,16,16).cuda(), torch.rand(1,256,2,8,8).cuda()]
    head=Detection_Head(
        num_features_in=256,
        num_classes=1,
        num_anchors=3,
        feature_size=256
        ).cuda()
    regressions, classifications = head(features)
    print(regressions.shape, classifications.shape)


