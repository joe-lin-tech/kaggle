import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights
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
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

        self.head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(7, 2, 2), stride=(5, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(5, 2, 2), stride=(3, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=(3, 2, 2), stride=(2, 1, 1))
        )

        self.out_bowel = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.out_extravasation = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.out_kidney = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.out_liver = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.out_spleen = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = x.view(b * (N_CHANNELS // 3), 3, x.shape[-2], x.shape[-1])
        x = self.backbone(x)
        x = x.view(b, N_CHANNELS // 3, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.head(x)
        x = torch.flatten(x, 1)
        out = {
            'bowel': self.out_bowel(x),
            'extravasation': self.out_extravasation(x),
            'kidney': self.out_kidney(x),
            'liver': self.out_liver(x),
            'spleen': self.out_spleen(x)
        }
        return out
