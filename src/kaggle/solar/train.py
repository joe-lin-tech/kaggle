from pathlib import Path

from jsonargparse import CLI
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from kaggle.solar.dataset import MAGFiLODataModule
from kaggle.solar.model import UNet


def train(
    output_dir: Path = Path("runs/solar"),
    num_epochs: int = 30,
    project: str = "magfilo-solar",
    name: str = "unet",
):
    datamodule = MAGFiLODataModule()
    seed_everything(datamodule.seed, workers=True)
    model = UNet()
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
        logger=WandbLogger(project=project, name=name, save_dir=output_dir),
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    CLI(train, as_positional=False)
