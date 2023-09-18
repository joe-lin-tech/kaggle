import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.ops import DropBlock3d
from params import *


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, out, labels):
        return self.loss(out, labels.long())

class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()

        # (512, 512, 1)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (256, 256, 64)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (128, 128, 128)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (64, 64, 256)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # (32, 32, 512)

        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) 
        self.decoder1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, N_CLASSES, kernel_size=1)
    
    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.pool1(x1)

        x2 = self.encoder2(x)
        x = self.pool2(x2)

        x3 = self.encoder3(x)
        x = self.pool3(x3)

        x4 = self.encoder4(x)
        x = self.pool4(x4)

        x = self.encoder5(x)

        x = self.upconv1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder2(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder3(x)

        x = self.upconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder4(x)

        out = self.out(x)
        return out


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.kidney = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        self.liver = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
        self.spleen = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(DEVICE))
    
    def forward(self, out, labels):
        kidney, liver, spleen = out
        ce_loss = self.kidney(kidney, labels[:, 2:5].float()) + self.liver(liver, labels[:, 5:8].float()) + self.spleen(spleen, labels[:, 8:11].float())

        kidney, liver, spleen = F.softmax(kidney, dim=-1), F.softmax(liver, dim=-1), F.softmax(spleen, dim=-1)
        healthy = torch.cat([kidney[:, 0:1], liver[:, 0:1], spleen[:, 0:1]], dim=-1)
        any_injury, _ = torch.max(1 - healthy, keepdim=True, dim=-1)
        any_injury = torch.clamp(any_injury, 1e-7, 1 - 1e-7)
        any_loss = torch.mean(torch.neg(6 * labels[:, 11:12] * torch.log(any_injury) + (1 - labels[:, 11:12]) * torch.log(1 - any_injury)))
        return ce_loss + any_loss
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv3d(2048, 256, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(256, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(128, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # nn.Dropout3d()
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
        # prob = self.slice_predictor(scans)
        # sliced_scans = torch.multiply(scans, torch.reshape(prob, (prob.shape[0], prob.shape[1], 1, 1)))

        x = torch.reshape(scans, (b * (c // 3), 3, h, w))
        x = self.backbone(x)
        x = torch.reshape(x, (b, c // 3, x.shape[-3], x.shape[-2], x.shape[-1])).transpose(1, 2)
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        x = self.out(x)
        # x = self.out(mask_features)
        kidney = self.out_kidney(x)
        liver = self.out_liver(x)
        spleen = self.out_spleen(x)
        return kidney, liver, spleen