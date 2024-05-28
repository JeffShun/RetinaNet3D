import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, anchor_path, pyramid_levels):
        super(Anchors, self).__init__()

        self.anchors = self.parser_anchor_txt(anchor_path)
        self.pyramid_levels = pyramid_levels

    def forward(self, image):
        device = image.device
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        self.image_shapes = [(image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 6)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):            
            # 将相对坐标转化为像素空间绝对坐标
            shifted_anchors = self.shift(self.image_shapes[idx], self.pyramid_levels[idx], self.anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = torch.from_numpy(all_anchors.astype(np.float32)).cuda(device).repeat(image.shape[0], 1, 1)
        return all_anchors


    def parser_anchor_txt(self, anchor_path):
        anchors_dwh = []
        with open(anchor_path, 'r') as f:
            for line in f:
                anchor = line.strip().split(",")
                anchor = list(map(lambda x: int(x), anchor))
                anchors_dwh.append(anchor)
        anchors_dwh = np.array(anchors_dwh)

        # convert to (z1, y1, x1, z2, y2, x2)
        anchors_zyx = np.zeros((anchors_dwh.shape[0], 6))
        anchors_zyx[:, 3:] = anchors_dwh
        anchors_zyx[:, 0::3] -= np.tile(anchors_zyx[:, 3] * 0.5, (2, 1)).T
        anchors_zyx[:, 1::3] -= np.tile(anchors_zyx[:, 4] * 0.5, (2, 1)).T   
        anchors_zyx[:, 2::3] -= np.tile(anchors_zyx[:, 5] * 0.5, (2, 1)).T   
        return anchors_zyx


    def shift(self, image_shape, pyramid_level, anchors):
        shift_z = (np.arange(0, image_shape[0]) + 0.5) * 2**pyramid_level
        shift_y = (np.arange(0, image_shape[1]) + 0.5) * 2**pyramid_level
        shift_x = (np.arange(0, image_shape[2]) + 0.5) * 2**pyramid_level

        shift_z, shift_y, shift_x = np.meshgrid(shift_z, shift_y, shift_x)

        shifts = np.vstack((shift_z.ravel(), shift_y.ravel(), shift_x.ravel(), shift_z.ravel(), shift_y.ravel(), shift_x.ravel())).transpose()

        # add A anchors (1, A, 6) to
        # cell K shifts (K, 1, 6) to get
        # shift anchors (K, A, 6)
        # reshape to (K*A, 6) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 6))

        return all_anchors


if __name__ == "__main__":
    img = torch.rand(2,1,64,256,256).cuda()
    anchor_generator=Anchors(
        anchor_path=r"F:\Code\Object_Detection3D\train\train_data\processed_data\anchors.txt",
        pyramid_levels=[3,4,5]
        )
    anchors = anchor_generator(img)
    print(anchors.shape)
    
    