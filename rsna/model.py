import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights
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
        any_injury = bowel + extravasation + kidney[:, 1:2] + kidney[:, 2:3] + liver[:, 1:2] + liver[:, 2:3] + spleen[:, 1:2] + liver[:, 2:3]
        any_injury /= 5
        any_loss = self.any(any_injury, labels[:, 5:6].float())

        return bce_loss + ce_loss + any_loss
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return F.relu(self.point_conv(self.depth_conv(x)))
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()
        self.conv1 = DepthwiseSeparableConv(N_CHANNELS, N_CHANNELS // 2, 9, stride=2)
        self.conv2 = DepthwiseSeparableConv(N_CHANNELS // 2, N_CHANNELS // 4, 11, stride=1)
        self.conv3 = nn.Conv2d(N_CHANNELS // 4, N_CHANNELS // 8, 11, stride=1)
        self.conv4 = nn.Conv2d(N_CHANNELS // 8, 3, 9, stride=1)

        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # self.backbone.conv_proj = nn.Conv2d(N_CHANNELS, 768, 16, stride=16)
        for layer in self.backbone.encoder.layers:
            layer.self_attention._reset_parameters()
            for param in layer.self_attention.parameters():
                param.requires_grad = True
        self.backbone.heads.head = nn.Linear(768, 384)

        self.linear_bowel = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 192))
        self.linear_extravasation = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 192))
        self.linear_kidney = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 192))
        self.linear_liver = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 192))
        self.linear_spleen = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 192))

        self.out_bowel = nn.Linear(192, 1)
        self.out_extravasation = nn.Linear(192, 1)
        self.out_kidney = nn.Linear(192, 3)
        self.out_liver = nn.Linear(192, 3)
        self.out_spleen = nn.Linear(192, 3)
    
    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = self.backbone(x)
        out = {
            'bowel': self.out_bowel(F.gelu(self.linear_bowel(x))),
            'extravasation': self.out_extravasation(F.gelu(self.linear_extravasation(x))),
            'kidney': self.out_kidney(F.gelu(self.linear_kidney(x))),
            'liver': self.out_liver(F.gelu(self.linear_liver(x))),
            'spleen': self.out_spleen(F.gelu(self.linear_spleen(x)))
        }
        return out
