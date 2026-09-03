from pathlib import Path

from jsonargparse import CLI
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from kaggle.solar.dataset import MAGFiLODataModule
from kaggle.solar.model import UNet


def train(output_dir: Path = Path("runs/solar"), num_epochs: int = 30):
    datamodule = MAGFiLODataModule()
    model = UNet()
    seed_everything(datamodule.seed, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="{epoch:02d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint, EarlyStopping(monitor="val_dice", mode="max", patience=8)],
        default_root_dir=output_dir,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    CLI(train, as_positional=False)
