import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights
from torchvision.ops import DropBlock2d, DropBlock3d
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
        kidney, liver, spleen = F.softmax(out['kidney'], dim=-1), F.softmax(out['liver'], dim=-1), F.softmax(out['spleen'], dim=-1)
        any_injury = torch.hstack([bowel, extravasation, kidney[:, 0:1], liver[:, 0:1], spleen[:, 0:1]])
        any_injury, _ = torch.max(1 - any_injury, keepdim=True, dim=-1)
        any_loss = self.any(any_injury, labels[:, 5:6].float())
        # any_loss = self.any(out['any'], labels[:, 5:6].float())

        return bce_loss + ce_loss + any_loss
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        for i, block in enumerate(self.backbone[4:]):
            self.backbone[i + 4] = nn.Sequential(block[0], DropBlock2d(p=0.2, block_size=3), block[1])
        # self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in list(self.backbone.children())[-1].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv3d(512, 384, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(384),
            nn.GELU(),
            nn.Dropout(0.4),
            DropBlock3d(p=0.4, block_size=3),
            nn.Conv3d(384, 256, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            # DropBlock3d(p=0.4, block_size=3),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            # DropBlock3d(p=0.4, block_size=3)
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.out_bowel = nn.Linear(64, 1)
        self.out_extravasation = nn.Linear(64, 1)
        self.out_kidney = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 3)
        )
        self.out_liver = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 3)
        )
        self.out_spleen = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 3)
        )
        # self.out_any = nn.Linear(64, 1)
    
    def forward(self, x):
        b = x.shape[0]
        c = x.shape[1]
        x = x.view(b * (c // 3), 3, x.shape[-2], x.shape[-1])
        x = self.backbone(x)
        x = x.view(b, c // 3, x.shape[-3], x.shape[-2], x.shape[-1]).transpose(1, 2)
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        out = {
            'bowel': self.out_bowel(x),
            'extravasation': self.out_extravasation(x),
            'kidney': self.out_kidney(x),
            'liver': self.out_liver(x),
            'spleen': self.out_spleen(x),
            # 'any': self.out_any(x)
        }
        return out