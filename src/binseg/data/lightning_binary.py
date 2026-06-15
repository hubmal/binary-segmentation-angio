import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from binseg.config import IMGSZ
from binseg.data.dataset import ICABinaryDataset, BinarySegmentationDataset
from binseg.data.preprocess import get_transforms


class ICABinaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        test_ratio: float = 0.1,
        img_size: tuple = IMGSZ,
        split=False,
        *args, **kwargs
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.img_size = img_size
        self.split = split
        self.args = args
        self.kwargs = kwargs
  
    def setup(self, stage):
        if not self.split:
            full_dataset_train = ICABinaryDataset(
                self.root, 
                transforms=get_transforms(self.img_size, train=True), 
                *self.args, **self.kwargs
            )
            full_dataset_val = ICABinaryDataset(
                self.root, 
                transforms=get_transforms(self.img_size, train=False), 
                *self.args, **self.kwargs
            )

            n_total = len(full_dataset_train)
            n_train = int(n_total * self.train_ratio)
            n_test = int(n_total * self.test_ratio)
            n_val = n_total - n_train - n_test

            indices = torch.randperm(n_total).tolist()

            self.train_dataset = Subset(full_dataset_train, indices[:n_train])
            self.val_dataset = Subset(full_dataset_val, indices[n_train : n_train + n_val])
            self.test_dataset = Subset(full_dataset_val, indices[n_train + n_val:])

        else:
            self.train_dataset = BinarySegmentationDataset(
                self.root, "train", transforms=get_transforms(self.img_size, train=True), *self.args, **self.kwargs
            )
            self.val_dataset = BinarySegmentationDataset(
                self.root, "val", transforms=get_transforms(self.img_size, train=False), *self.args, **self.kwargs
            )
            self.test_dataset = BinarySegmentationDataset(
                self.root, "test", transforms=get_transforms(self.img_size, train=False), *self.args, **self.kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

def get_datamodule(*args, **kwargs):
    return ICABinaryDataModule(*args, **kwargs)