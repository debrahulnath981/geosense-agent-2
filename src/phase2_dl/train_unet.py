import numpy as np
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
NUM_CLASSES = 5
IN_CHANNELS = 6

LEARNING_RATE = 0.001


# ============================================================
# DATASET
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(self):

        images_2015 = np.load(
            "data/satellite/chips/chips_2015_clean.npy"
        )

        labels_2015 = np.load(
            "data/satellite/chips/label_chips_2015.npy"
        )

        images_2023 = np.load(
            "data/satellite/chips/chips_2023_clean.npy"
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

        # Change:
        # (224, 224, 6)
        # to:
        # (6, 224, 224)

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Starting U-Net training setup...\n")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SatelliteDataset()

    print("Total samples:", len(dataset))

    # --------------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------------

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------------
    # U-Net model
    # --------------------------------------------------------

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    model = model.to(device)

    print("\nU-Net model created.")
    print("Input channels:", IN_CHANNELS)
    print("Output classes:", NUM_CLASSES)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------
    # ========================================================
    # TRAINING
    # ========================================================

    NUM_EPOCHS = 10

    best_val_loss = float("inf")

    train_history = []
    val_history = []

    print("\nStarting U-Net training...")
    print("Epochs:", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_pixels += (
                predictions == labels
            ).sum().item()

            total_pixels += labels.numel()

        train_loss = (
            running_loss / len(train_loader)
        )

        train_accuracy = (
            correct_pixels / total_pixels
        ) * 100

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss_total += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.numel()

        val_loss = (
            val_loss_total / len(val_loader)
        )

        val_accuracy = (
            val_correct / val_total
        ) * 100

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "models/saved/unet_best.pth"
            )

            print("  Best model saved.")

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        "models/saved/unet_final.pth"
    )

    print("\nFinal model saved.")

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    import pandas as pd

    history_df = pd.DataFrame({
        "epoch": range(1, NUM_EPOCHS + 1),
        "train_loss": train_history,
        "val_loss": val_history
    })

    history_df.to_csv(
        "outputs/reports/unet_training_history.csv",
        index=False
    )

    print(
        "Training history saved."
    )

    print("\nU-Net training completed!")