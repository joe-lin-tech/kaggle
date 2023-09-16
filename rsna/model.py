import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.ops import DropBlock3d
from params import *


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        
        # backbone = resnet18(ResNet18_Weights.DEFAULT)
        self.backbone = vit_b_16(ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Sequential()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.conv_proj.parameters():
            param.requires_grad = True
        for layer in self.backbone.encoder.layers:
            for param in layer.self_attention.parameters():
                param.requires_grad = True
        # for param in self.backbone.encoder.layers.encoder_layer_10.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.encoder.layers.encoder_layer_11.parameters():
        #     param.requires_grad = True
        # self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.backbone[-1].parameters():
        #     param.requires_grad = True

        self.fcn = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout()
        )

    def forward(self, masked_scans):
        b, c, h, w = masked_scans.shape
        x = torch.reshape(masked_scans, (b * (c // 3), 3, h, w))
        x = self.backbone(resize(x, (224, 224)))
        # x = self.backbone(x)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x = torch.flatten(x, 1)
        x = self.fcn(x)
        x = torch.reshape(x, (b, c // 3, -1))
        x = torch.transpose(x, 1, 2)
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        return x
    

class SlicePredictor(nn.Module):
    def __init__(self):
        super(SlicePredictor, self).__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        self.layer_norm = nn.LayerNorm(512)
        self.pos_embedding = nn.Parameter(torch.randn(N_CHANNELS // 3, 512))

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.linear = nn.Linear(512, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * (c // 3), 3, h, w)
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = torch.reshape(x, (b, (c // 3), 512))
        x = self.layer_norm(x) + self.pos_embedding
        x = self.encoder(x)
        x = torch.squeeze(self.linear(x), dim=-1)
        x = torch.sigmoid(x)
        x = torch.squeeze(F.interpolate(x.unsqueeze(1), size=c, mode='linear'), dim=1)
        return x


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # self.bowel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).to(DEVICE))
        # self.extravasation = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(DEVICE))
        self.kidney = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        self.liver = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        self.spleen = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        # self.any = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(DEVICE))
    
    def forward(self, out, labels):
        # bce_loss = self.bowel(out['bowel'], labels[:, 0:1].float()) + self.extravasation(out['extravasation'], labels[:, 1:2].float())
        ce_loss = self.kidney(out['kidney'], labels[:, 2:5].float()) + self.liver(out['liver'], labels[:, 5:8].float()) + self.spleen(out['spleen'], labels[:, 8:11].float())
        
        # bowel, extravasation = torch.sigmoid(out['bowel']), torch.sigmoid(out['extravasation'])
        # kidney, liver, spleen = F.softmax(out['kidney'], dim=-1), F.softmax(out['liver'], dim=-1), F.softmax(out['spleen'], dim=-1)
        # healthy = torch.hstack([1 - bowel, 1 - extravasation, kidney[:, 0:1], liver[:, 0:1], spleen[:, 0:1]])
        # any_injury, _ = torch.max(1 - healthy, keepdim=True, dim=-1)
        # any_loss = self.any(any_injury, labels[:, 5:6].float())

        # return bce_loss + ce_loss + any_loss
        return ce_loss
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()

        # self.mask_encoder = MaskEncoder()
        self.slice_predictor = SlicePredictor()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv3d(2048, 512, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(512, 256, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(256, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d()
            # nn.Dropout(0.4)
        )

        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Dropout()
        )

        # self.out_bowel = nn.Linear(32, 1)
        # self.out_extravasation = nn.Linear(32, 1)
        self.out_kidney = nn.Linear(64, 3)
        self.out_liver = nn.Linear(64, 3)
        self.out_spleen = nn.Linear(64, 3)
    
    def forward(self, scans):
        b, c, h, w = scans.shape
        # mask_features = self.mask_encoder(masked_scans)
        prob = self.slice_predictor(scans)
        for _ in range(b):
            indices = torch.where(prob[_] > 0.5)[0]
            if indices.shape[0] > 0:
                scans[_] = torch.squeeze(F.interpolate(
                    torch.unsqueeze(torch.unsqueeze(torch.index_select(scans, 1, indices), dim=0), dim=0),
                    size=(c, h, w), mode='trilinear'
                ), dim=(0, 1))

        x = torch.reshape(scans, (b * (c // 3), 3, h, w))
        x = self.backbone(x)
        x = torch.reshape(x, (b, c // 3, x.shape[-3], x.shape[-2], x.shape[-1])).transpose(1, 2)
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        # x = torch.cat([x, mask_features], dim=1)
        x = self.out(x)
        # x = self.out(mask_features)
        out = {
            # 'bowel': self.out_bowel(x),
            # 'extravasation': self.out_extravasation(x),
            'kidney': self.out_kidney(x),
            'liver': self.out_liver(x),
            'spleen': self.out_spleen(x)
        }
        return out