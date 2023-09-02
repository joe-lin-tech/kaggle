import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from params import *

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bowel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).to(DEVICE))
        self.extravasation = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(DEVICE))
        self.organ = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        self.any = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(DEVICE))
    
    def forward(self, out, labels):
        bce_loss = self.bowel(out['bowel'], labels[:, 0:1].float()) + self.extravasation(out['extravasation'], labels[:, 1:2].float())
        ce_loss = self.organ(out['kidney'], labels[:, 2]) + self.organ(out['liver'], labels[:, 3]) + self.organ(out['spleen'], labels[:, 4])
        
        bowel, extravasation = torch.sigmoid(out['bowel']), torch.sigmoid(out['extravasation'])
        kidney, liver, spleen = F.softmax(out['kidney']), F.softmax(out['liver']), F.softmax(out['spleen'])
        any_injury = torch.hstack([bowel, extravasation, kidney[:, 0:1], liver[:, 0:1], spleen])
        any_injury, _ = torch.max(any_injury, keepdim=True, dim=-1)
        any_loss = self.any(any_injury, labels[:, 5:6].float())

        return bce_loss + ce_loss + any_loss
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.batch_norm(F.relu(self.point_conv(self.depth_conv(x))))
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()
        self.conv1 = DepthwiseSeparableConv(96, 48, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(48, 3, 5, dilation=3)

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.out_bowel = nn.Linear(1280, 1)
        self.out_extravasation = nn.Linear(1280, 1)
        self.out_kidney = nn.Linear(1280, 3)
        self.out_liver = nn.Linear(1280, 3)
        self.out_spleen = nn.Linear(1280, 3)
    
    def forward(self, x):
        x = F.relu(self.conv2(self.conv1(x)))
        x = self.backbone(x)
        out = {
            'bowel': self.out_bowel(x),
            'extravasation': self.out_extravasation(x),
            'kidney': self.out_kidney(x),
            'liver': self.out_liver(x),
            'spleen': self.out_spleen(x)
        }
        return out
