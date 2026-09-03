import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet(LightningModule):
    def __init__(self, num_channels: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.enc1, self.enc2, self.enc3 = (
            ConvBlock(1, num_channels),
            ConvBlock(num_channels, 2 * num_channels),
            ConvBlock(2 * num_channels, 4 * num_channels),
        )
        self.neck = ConvBlock(4 * num_channels, 8 * num_channels)
        self.pool = nn.MaxPool2d(2)
        self.up3, self.dec3 = (
            nn.ConvTranspose2d(8 * num_channels, 4 * num_channels, kernel_size=2, stride=2),
            ConvBlock(8 * num_channels, 4 * num_channels),
        )
        self.up2, self.dec2 = (
            nn.ConvTranspose2d(4 * num_channels, 2 * num_channels, kernel_size=2, stride=2),
            ConvBlock(4 * num_channels, 2 * num_channels),
        )
        self.up1, self.dec1 = (
            nn.ConvTranspose2d(2 * num_channels, num_channels, kernel_size=2, stride=2),
            ConvBlock(2 * num_channels, num_channels),
        )
        self.head = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.neck(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.head(d1)

    @staticmethod
    def dice_score(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
        pred = (torch.sigmoid(logits) >= 0.5).float()
        intersect = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return ((2 * intersect + eps) / (union + eps)).mean()

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        prob = torch.sigmoid(logits)
        intersect = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersect + 1e-6) / (union + 1e-6)).mean()
        return 0.5 * self.bce(logits, target) + 0.5 * dice_loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["image"])
        loss = self.loss(logits, batch["mask"])
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.loss(logits, batch["mask"])
        dice = self.dice_score(logits, batch["mask"])
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )
        self.log(
            "val_dice",
            dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
