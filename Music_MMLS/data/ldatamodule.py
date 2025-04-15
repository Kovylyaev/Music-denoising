import lightning as L
from pathlib import Path
from torch.utils.data import random_split, DataLoader

from Music_MMLS.data.download import download_data
from Music_MMLS.data.dataset import Music_Dataset


class MusicDataModule(L.LightningDataModule):
    def __init__(self, cfg, batch_size):
        super().__init__()
        self.data_dir = Path(cfg.data_dir)
        self.clean_data_dir = Path(cfg.clean_dir)
        self.noise_data_dir = Path(cfg.noise_dir)
        self.dataset_size = cfg.size
        self.test_size = cfg.test_size
        self.batch_size = batch_size
        # self.transform

    def prepare_data(self):
        # download
        download_data(dataset_path="andradaolteanu/gtzan-dataset-music-genre-classification", out_path=self.clean_data_dir, dataset_path_add='Data/genres_original')
        download_data(dataset_path="mlneo07/random-noise-audio", out_path=self.noise_data_dir, audio_duration=30)

    def setup(self, stage: str):
        # Assign train/test datasets for use in dataloaders
        train_clean_data, test_clean_data = random_split(list(self.clean_data_dir.iterdir()), [1 - self.test_size, self.test_size])
        train_noise_data, test_noise_data = random_split(list(self.noise_data_dir.iterdir()), [1 - self.test_size, self.test_size])

        self.train_dataset = Music_Dataset(size=self.dataset_size, clean_files=train_clean_data, noise_files=train_noise_data)
        self.test_dataset = Music_Dataset(size=self.dataset_size, clean_files=test_clean_data, noise_files=test_noise_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
