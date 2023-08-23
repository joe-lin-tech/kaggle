import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, out, labels):
        bce_loss = 2 * self.bce(out['bowel'], labels[:, 0:1].float()) + 6 * self.bce(out['extravasation'], labels[:, 1:2].float())
        ce_loss = self.ce(out['kidney'], labels[:, 2]) + self.ce(out['liver'], labels[:, 3]) + self.ce(out['spleen'], labels[:, 4])
        return bce_loss + ce_loss


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()
        # (96, 512, 512)
        self.conv1 = DepthwiseSeparableConv(96, 128, 7, stride=2, padding=3) # (128, 256, 256)
        self.conv2 = nn.Conv2d(128, 96, 5, dilation=3) # (96, 244, 244)

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone_features = backbone.features[2:]
        self.backbone_features[0][0].block = self.backbone_features[0][0].block[1:]
        self.backbone_features.eval()
        self.backbone_pool = backbone.avgpool
        self.backbone_pool.eval()

        self.out_bowel = nn.Linear(1280, 1)
        self.out_extravasation = nn.Linear(1280, 1)
        self.out_kidney = nn.Linear(1280, 3)
        self.out_liver = nn.Linear(1280, 3)
        self.out_spleen = nn.Linear(1280, 3)
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.backbone_pool(self.backbone_features(x))

        x = torch.reshape(x, (x.shape[0], -1))
        out = {
            'bowel': self.out_bowel(x),
            'extravasation': self.out_extravasation(x),
            'kidney': self.out_kidney(x),
            'liver': self.out_liver(x),
            'spleen': self.out_spleen(x)
        }
        return out