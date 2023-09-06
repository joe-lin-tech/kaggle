import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights
from torchvision.ops import DropBlock2d, DropBlock3d
from SAM_Med2D.segment_anything import sam_model_registry
from SAM_Med2D.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from params import *


class MaskPredictor(nn.Module):
    def __init__(self):
        super(MaskPredictor, self).__init__()
        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = MASK_MODEL
        model = sam_model_registry["vit_b"](args).to(DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(model, pred_iou_thresh=0.4, stability_score_thresh=0.8)

    def forward(self, x):
        for b in range(x.shape[0]):
            size = 12
            for i in range(size // 2, N_CHANNELS - (size // 2), size):
                image = x[b, i - 1:i + 2, :, :].transpose(0, 1).transpose(1, 2)
                # plt.imshow(image[:, :, 1], cmap='bone')
                masks = self.mask_generator.generate(image)
                mask = np.where(np.logical_or.reduce([mask['segmentation'] for mask in masks]), 1, 0.2)
                x[b, i - (size // 2):i + (size // 2), :, :] *= mask
                # self.show_mask(mask, plt.gca())
                # for mask in masks:
                #     self.show_mask(mask['segmentation'], plt.gca())
                # plt.axis('off')
                # plt.show()

    def show_mask(self, mask, ax):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


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

        # self.mask_predictor = MaskPredictor()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        # for i, block in enumerate(self.backbone[4:]):
        #     self.backbone[i + 4] = nn.Sequential(block[0], DropBlock2d(p=0.2, block_size=3), block[1])
        # self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
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
        self.mask_predictor(x)
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