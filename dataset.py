import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2

class SignalDataset(Dataset):
    dataset_path = "./data/original_data.csv"

    def __init__(self, width: int, stride: int, number_of_frames: int):
        super().__init__()
        self.width = width
        self.stride = stride
        self.number_of_frames = number_of_frames

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

        a = self.width
        b = self.stride
        c = self.number_of_frames

        pattern_length = a * b
        n_patterns = round(len(df) / c) - 10

        start = 0
        end = a * b
        X_out, y_out = list(), list()

        for _ in range(n_patterns):
            slide_data = df[start:end]
            start += c
            end += c
            frames = []
            for i in range(b):
                plt.plot(range(len(slide_data[:(i + 1) * a])), slide_data[:(i + 1) * a])
                plt.ylim(0, 2)
                plt.xlim(0, b * a)
                plt.axis('off')
                plt.savefig('glove/{}.png'.format(i), transparent=True, bbox_inches='tight', pad_inches=0)
                plt.show()
                img = cv2.imread('glove/{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                img[img == 255] = 0
                img[img > 0] = 1
                #         plt.imshow(img)
                frames.append(img)

            X_out.append(frames)
            y_out.append(label.iloc[pattern_length + _ * c])

        X = np.array(X_out).reshape(len(X_out), b, img_size, img_size, 1)
        y = np.array(y_out).reshape(len(y_out), 1)

        X, y = shuffle(X, y)


        self.input = X
        self.target = y

    def __len__(self):
        return round(len(self.target) / self.stride) - self.width

    def __getitem__(self, index):
        x = self.input
        y = self.target

        return x, y


class SignalDataModule(LightningDataModule):
    def __init__(self, width: int, stride: int, number_of_frames: int, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        dataset = SignalDataset(width, stride, number_of_frames)
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
