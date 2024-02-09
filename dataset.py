import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class SignalDataset(Dataset):
    dataset_path = "./data/original_data.csv"

    def __init__(self, width: int, stride: int):
        super().__init__()
        self.width = width
        self.stride = stride

        df = pd.read_csv(SignalDataset.dataset_path)

        df.reset_index(drop=True, inplace=True)

        normalization_list = [
            "Middle Knuckle",
            "Index Knuckle",
            "Thumb Knuckle",
            "Index Finger",
            "Middle Finger",
            "Thumb Finger",
            "Palm",
        ]
        for channel in normalization_list:
            df[channel] = (df[channel] - df[channel].min()) / (
                df[channel].max() - df[channel].min()
            )

        X_col = [
            "Middle Knuckle",
            "Index Knuckle",
            "Thumb Knuckle",
            "Index Finger",
            "Middle Finger",
            "Thumb Finger",
            "Palm",
        ]
        self.input = df[X_col]
        self.target = df["Label"]

    def __len__(self):
        return round(len(self.target) / self.stride) - self.width

    def __getitem__(self, index):
        idx = index * self.stride
        x = self.input[idx : idx + self.width]
        y = self.target[idx : idx + self.width]

        return x, y


class SignalDataModule(LightningDataModule):
    def __init__(self, width: int, seq_length: int, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        dataset = SignalDataset(width, seq_length)
        length = len(dataset)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [int(0.8 * length), length - int(0.8 * length)]
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )


if __name__ == "__main__":
    pass
