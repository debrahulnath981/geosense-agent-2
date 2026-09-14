import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SatelliteDataset(Dataset):

    def __init__(self):

        images_2015 = np.load(
            "data/satellite/chips/chips_2015.npy"
        )

        labels_2015 = np.load(
            "data/satellite/chips/label_chips_2015.npy"
        )

        images_2023 = np.load(
            "data/satellite/chips/chips_2023.npy"
        )

        labels_2023 = np.load(
            "data/satellite/chips/label_chips_2023.npy"
        )

        self.images = np.concatenate(
            [images_2015, images_2023],
            axis=0
        )

        self.labels = np.concatenate(
            [labels_2015, labels_2023],
            axis=0
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        # (224, 224, 6) → (6, 224, 224)
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":

    dataset = SatelliteDataset()

    print("Total samples:", len(dataset))

    # 80% training / 20% validation
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    print("Training samples:", len(train_indices))
    print("Validation samples:", len(val_indices))

    # Create subsets
    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Test one batch
    images, labels = next(iter(train_loader))

    print("\nBatch test:")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("\nDataLoader test successful!")