import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


class MAGFiLODataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: Literal["train", "val", "test"] = "train",
        input_size: tuple[int, int] = (512, 512),
        records: list[dict] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size

        if split != "test":
            annotation_path = (
                self.root_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
            )
            with open(annotation_path, encoding="utf-8") as f:
                annotations = json.load(f)
            labels_by_image = defaultdict(list)
            for annotation in annotations["annotations"]:
                labels_by_image[str(annotation["image_id"])].append(annotation)
            self.labels_by_image = dict(labels_by_image)
            self.records = list(records) if records is not None else annotations["images"]
            self.image_dir = self.root_dir / "train" / "train_images"
        else:
            self.labels_by_image = {}
            self.records = [
                {"id": path.stem, "file_name": path.name, "height": 2048, "width": 2048}
                for path in sorted((self.root_dir / "test" / "test_images").glob("*"))
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            self.image_dir = self.root_dir / "test" / "test_images"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_id = str(record["id"])
        image_path = self.image_dir / record["file_name"]
        target_h, target_w = self.input_size

        with Image.open(image_path) as opened:
            image = opened.convert("L").resize((target_w, target_h), Image.Resampling.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        sample = {
            "image_id": image_id,
            "image": torch.from_numpy(image).unsqueeze(0),
            "file_name": record["file_name"],
        }
        if self.split != "test":
            mask = self.rasterize(record)
            sample["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        return sample

    def rasterize(self, record: dict[str, Any]) -> Image.Image:
        orig_h, orig_w = record["height"], record["width"]
        tgt_h, tgt_w = self.input_size
        mask = Image.new("L", (tgt_w, tgt_h), 0)
        drawer = ImageDraw.Draw(mask)
        sx, sy = tgt_w / orig_w, tgt_h / orig_h

        for annotation in self.labels_by_image.get(str(record["id"]), []):
            for seg in annotation["segmentation"]:
                if len(seg) < 6:
                    continue
                points = [(seg[i] * sx, seg[i + 1] * sy) for i in range(0, len(seg) - 1, 2)]
                drawer.polygon(points, fill=1)
        return mask


class MAGFiLODataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: Path = Path("data/MAGFiLO_1.0_Kaggle_2026"),
        batch_size: int = 8,
        input_size: tuple[int, int] = (512, 512),
        val_split: float = 0.15,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        assert val_split > 0 and val_split < 1
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset: MAGFiLODataset | None = None
        self.val_dataset: MAGFiLODataset | None = None
        self.test_dataset: MAGFiLODataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate") and self.train_dataset is None:
            dataset = MAGFiLODataset(self.root_dir, split="train", input_size=self.input_size)
            names = sorted({record["file_name"] for record in dataset.records})
            random.Random(self.seed).shuffle(names)
            val_size = max(1, round(len(names) * self.val_split))
            val_names = set(names[:val_size])
            train_records = [
                record for record in dataset.records if record["file_name"] not in val_names
            ]
            val_records = [record for record in dataset.records if record["file_name"] in val_names]
            self.train_dataset = MAGFiLODataset(
                self.root_dir, split="train", input_size=self.input_size, records=train_records
            )
            self.val_dataset = MAGFiLODataset(
                self.root_dir, split="val", input_size=self.input_size, records=val_records
            )
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = MAGFiLODataset(
                self.root_dir, split="test", input_size=self.input_size
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
