import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.models import resnet50, resnet18, ResNet18_Weights
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
    

class SlicePredictor(nn.Module):
    def __init__(self):
        super(SlicePredictor, self).__init__()
        self.backbone = resnet18()

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
        self.kidney = nn.CrossEntropyLoss(reduction='none')
        self.liver = nn.CrossEntropyLoss(reduction='none')
        self.spleen = nn.CrossEntropyLoss(reduction='none')
        self.organ_weights = torch.tensor([[1.0, 2.0, 4.0]]).T.to(DEVICE)
        self.any_weights = torch.tensor([[6.0]]).to(DEVICE)
        # self.kidney = nn.CrossEntropyLoss()
        # self.liver = nn.CrossEntropyLoss()
        # self.spleen = nn.CrossEntropyLoss()
    
    def forward(self, out, labels):
        # kidney, liver, spleen = out
        # ce_loss = self.kidney(kidney, labels[:, 2:5].float()) + self.liver(liver, labels[:, 5:8].float()) + self.spleen(spleen, labels[:, 8:11].float())
        # return ce_loss
        kidney, liver, spleen = out
        ce_loss = self.kidney(kidney, labels[:, 2:5].float()) * (labels[:, 2:5].float() @ self.organ_weights) \
            + self.liver(liver, labels[:, 5:8].float()) * (labels[:, 5:8].float() @ self.organ_weights) \
                + self.spleen(spleen, labels[:, 8:11].float()) * (labels[:, 8:11].float() @ self.organ_weights)

        kidney, liver, spleen = F.softmax(kidney, dim=-1), F.softmax(liver, dim=-1), F.softmax(spleen, dim=-1)
        healthy = torch.cat([kidney[:, 0:1], liver[:, 0:1], spleen[:, 0:1]], dim=-1)
        any_injury, _ = torch.max(1 - healthy, keepdim=True, dim=-1)
        any_injury = torch.clamp(any_injury, 1e-7, 1 - 1e-7)
        any_loss = torch.neg(labels[:, 11:12] * torch.log(any_injury) + (1 - labels[:, 11:12]) * torch.log(1 - any_injury)) * (labels[:, 11:12].float() @ self.any_weights + 1)
        # any_loss = torch.mean(any_loss, dim=0)
        return torch.mean(ce_loss + any_loss, dim=0)
    

class TraumaDetector(nn.Module):
    def __init__(self):
        super(TraumaDetector, self).__init__()

        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        
        self.head = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(0.3),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(256, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.3),
            # DropBlock3d(p=0.5, block_size=3),
            nn.Conv3d(128, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.out_kidney = nn.Linear(64, 3)
        self.out_liver = nn.Linear(64, 3)
        self.out_spleen = nn.Linear(64, 3)
    
    def forward(self, scans: Tensor):
        b, c, h, w = scans.shape

        x = scans.view(b * (c // 3), 3, h, w)
        x = self.backbone(x)
        x = x.reshape(b, c // 3, x.shape[-3], x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.reshape(b, -1)
        kidney = self.out_kidney(x)
        liver = self.out_liver(x)
        spleen = self.out_spleen(x)

        return kidney, liver, spleen