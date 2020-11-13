from models.ResNet_50 import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


class NADS_Net(nn.Module):
    def __init__(self, resnet_pretrained=True):
        super(NADS_Net, self).__init__()
        self.resnet50 = resnet50()

        self.relu = nn.ReLU(inplace=True)
        # FPN layers
        # reduce channels
        self.lateral_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_2 = nn.Conv2d(256, 256, kernel_size=1)
        # reduce upsampling bias
        self.smooth_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # upsampling FPN layers to fixed size
        self.concat_5 = self._concat_block()
        self.concat_4 = self._concat_block()
        self.concat_3 = self._concat_block()
        self.concat_2 = self._concat_block()

        # confidence maps and PAFs(part affinity fields)
        # the number of keypoints detection head out_channel is equal to
        # keypoints' number in training dataset + 1(background) + 1(neck)
        self.keypoint_heatmaps = self._heatmap_block(17 + 1 + 1)
        # the number of PAFs detection head out_channel is twice as the number of keypoints channel
        self.PAF_heatmaps = self._heatmap_block(17 * 2 + 2 + 2)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # init pretrained resnet-50 model params
        if resnet_pretrained:
            resnet_50_pretrained = model_zoo.load_url(model_urls['resnet50'])
            model_dict = self.resnet50.state_dict()
            resnet_50_pretrained = {k: v for k, v in resnet_50_pretrained.items() if k in model_dict}
            model_dict.update(resnet_50_pretrained)
            self.resnet50.load_state_dict(model_dict)

    def _concat_block(self, scale_factor=1):
        # in NADS-Net paper, the channel number of concated feature maps is 512,
        # i.e. channel number of p2-p5 is downsampling from 256 to 128 then concat
        layers = [
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1)
        ]
        return nn.Sequential(*layers)

    def _concat_upsample(self, x, H, W):
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)

    def _heatmap_block(self, out_channels):
        layers = [
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        ]
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        # ResNet_50 outputs feature maps in 4 levels
        c2, c3, c4, c5 = self.resnet50(x)

        # FPN - upsampling and lateral connection
        p5 = self.relu(self.lateral_5(c5))
        p4 = self.relu(self._upsample_add(p5, self.lateral_4(c4)))
        p3 = self.relu(self._upsample_add(p4, self.lateral_3(c3)))
        p2 = self.relu(self._upsample_add(p3, self.lateral_2(c2)))

        # FPN - reducing upsampling bias
        p4 = self.relu(self.smooth_1(p4))
        p3 = self.relu(self.smooth_1(p3))
        p2 = self.relu(self.smooth_1(p2))

        concat_upsample_h, concat_upsample_w = p2.shape[2], p2.shape[3]
        concated_feature_maps = torch.cat([
            self._concat_upsample(self.relu(self.concat_5(p5)), concat_upsample_h, concat_upsample_w),
            self._concat_upsample(self.relu(self.concat_4(p4)), concat_upsample_h, concat_upsample_w),
            self._concat_upsample(self.relu(self.concat_3(p3)), concat_upsample_h, concat_upsample_w),
            self.relu(self.concat_2(p2))], dim=1)

        keypoint_heatmaps = self.keypoint_heatmaps(concated_feature_maps)
        PAF_heatmaps = self.PAF_heatmaps(concated_feature_maps)

        return [keypoint_heatmaps], [PAF_heatmaps]
