import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import Music_Dataset
from .sampler import AccedingSequenceLengthBatchSampler
from .collate_fn import collate_fn
from .download import download_data

class MusicDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        size: int = 1000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.size = size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        download_data(self.data_dir)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            clean_files = [os.path.join(self.data_dir, "clean", f) for f in os.listdir(os.path.join(self.data_dir, "clean"))]
            noise_files = [os.path.join(self.data_dir, "noise", f) for f in os.listdir(os.path.join(self.data_dir, "noise"))]
            
            # Split files for train and validation
            train_size = int(0.8 * len(clean_files))
            train_clean = clean_files[:train_size]
            val_clean = clean_files[train_size:]
            
            train_noise = noise_files[:train_size]
            val_noise = noise_files[train_size:]
            
            self.train_dataset = Music_Dataset(self.size, train_clean, train_noise)
            self.val_dataset = Music_Dataset(self.size // 5, val_clean, val_noise)

        if stage == "test" or stage is None:
            clean_files = [os.path.join(self.data_dir, "clean", f) for f in os.listdir(os.path.join(self.data_dir, "clean"))]
            noise_files = [os.path.join(self.data_dir, "noise", f) for f in os.listdir(os.path.join(self.data_dir, "noise"))]
            self.test_dataset = Music_Dataset(self.size // 10, clean_files, noise_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=AccedingSequenceLengthBatchSampler(self.train_dataset, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=AccedingSequenceLengthBatchSampler(self.val_dataset, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_sampler=AccedingSequenceLengthBatchSampler(self.test_dataset, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
