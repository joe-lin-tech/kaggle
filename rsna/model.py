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
        
        bowel, extravasation = F.sigmoid(out['bowel']), F.sigmoid(out['extravasation'])
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
        return self.point_conv(self.depth_conv(x))


class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()
        # self.conv1 = DepthwiseSeparableConv(96, 128, 7, stride=2, padding=3)
        # self.dropout = nn.Dropout2d()
        # self.conv2 = nn.Conv2d(128, 64, 7, padding=2, dilation=6)
        self.conv1 = nn.Conv2d(3, 8, 9, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 11, stride=1)
        self.conv3 = nn.Conv2d(16, 8, 11, stride=1)
        self.conv4 = nn.Conv2d(8, 3, 9, stride=1)

        # backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone = backbone
        self.backbone.eval()
        # self.backbone_proj = backbone.conv_proj
        # self.cls_token = nn.Parameter(torch.rand(1, 1, 768))
        # self.backbone_encoder = backbone.encoder
        # self.backbone_encoder.eval()
        for layer in self.backbone.encoder.layers:
            layer.self_attention._reset_parameters()
            layer.self_attention.train()
        self.backbone.heads.head = nn.Linear(768, 384)
        # self.backbone_encoder.layers.encoder_layer_10.train()
        # self.backbone_encoder.layers.encoder_layer_10.self_attention._reset_parameters()
        # self.backbone_encoder.layers.encoder_layer_11.train()
        # self.backbone_encoder.layers.encoder_layer_11.self_attention._reset_parameters()

        # self.backbone_features = backbone.features[2:]
        # self.backbone_features[0][0].block = self.backbone_features[0][0].block[1:]
        # self.backbone_features.eval()
        # self.backbone_pool = backbone.avgpool
        # self.backbone_pool.eval()
        
        # self.linear = nn.Linear(768, 384)

        self.linear_bowel = nn.Linear(384, 192)
        self.linear_extravasation = nn.Linear(384, 192)
        self.linear_kidney = nn.Linear(384, 192)
        self.linear_liver = nn.Linear(384, 192)
        self.linear_spleen = nn.Linear(384, 192)

        self.out_bowel = nn.Linear(192, 1)
        self.out_extravasation = nn.Linear(192, 1)
        self.out_kidney = nn.Linear(192, 3)
        self.out_liver = nn.Linear(192, 3)
        self.out_spleen = nn.Linear(192, 3)
    
    def forward(self, x):
        # x = F.gelu(self.conv2(self.dropout(F.relu(self.conv1(x)))))
        # x = self.backbone_pool(self.backbone_features(x))
        # patches = self.img_to_patch(x, 16)
        x = F.relu(self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))))
        # proj = self.backbone_proj(x).flatten(start_dim=2).transpose(1, 2)
        # cls_token = self.cls_token.repeat(proj.shape[0], 1, 1) # fix cls_token training
        # proj = torch.cat([cls_token, proj], dim=1)
        # x = self.backbone_encoder(proj)
        x = self.backbone(x)
        # x = self.backbone_encoder(self.backbone_proj(x))
        
        # x = F.gelu(self.linear(x[:, 0, :]))
        # x = torch.reshape(x, (x.shape[0], -1))

        out = {
            'bowel': self.out_bowel(F.gelu(self.linear_bowel(x))),
            'extravasation': self.out_extravasation(F.gelu(self.linear_extravasation(x))),
            'kidney': self.out_kidney(F.gelu(self.linear_kidney(x))),
            'liver': self.out_liver(F.gelu(self.linear_liver(x))),
            'spleen': self.out_spleen(F.gelu(self.linear_spleen(x)))
        }
        return out
    
    def img_to_patch(self, x, patch_size, flatten_channels=True):
        """
        Inputs:
            x - Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                            as a feature vector instead of a image grid.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H' * W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H' * W', C * p_H * p_W]
        return x
