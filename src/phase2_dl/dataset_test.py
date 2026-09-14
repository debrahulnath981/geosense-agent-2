import numpy as np
import torch
from torch.utils.data import Dataset


class SatelliteDataset(Dataset):

    def __init__(self):

        # Load 2015 data
        images_2015 = np.load(
            "data/satellite/chips/chips_2015.npy"
        )

        labels_2015 = np.load(
            "data/satellite/chips/label_chips_2015.npy"
        )

        # Load 2023 data
        images_2023 = np.load(
            "data/satellite/chips/chips_2023.npy"
        )

        labels_2023 = np.load(
            "data/satellite/chips/label_chips_2023.npy"
        )

        # Combine both years
        self.images = np.concatenate(
            [images_2015, images_2023],
            axis=0
        )

        self.labels = np.concatenate(
            [labels_2015, labels_2023],
            axis=0
        )

        print("Images shape:", self.images.shape)
        print("Labels shape:", self.labels.shape)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        # Change image from:
        # (224, 224, 6)
        # to:
        # (6, 224, 224)

        image = np.transpose(image, (2, 0, 1))

        # Convert to PyTorch tensors
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


if __name__ == "__main__":

    dataset = SatelliteDataset()

    print("\nDataset size:", len(dataset))

    # Test first sample
    image, label = dataset[0]

    print("Image tensor shape:", image.shape)
    print("Label tensor shape:", label.shape)

    print("Image data type:", image.dtype)
    print("Label data type:", label.dtype)

    print("\nDataset test successful!")