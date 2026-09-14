import numpy as np
from pathlib import Path

CHIP_DIR = Path("data/satellite/chips")
LABEL_DIR = Path("data/satellite/labels")

CHIP_SIZE = 224


def create_label_chips(year):

    print(f"\nCreating label chips for {year}...")

    # Full label mask
    labels = np.load(LABEL_DIR / f"labels_{year}.npy")

    # Chip positions
    positions = np.load(CHIP_DIR / f"positions_{year}.npy")

    print("Full label shape:", labels.shape)
    print("Number of chip positions:", len(positions))

    label_chips = []

    for row, col in positions:

        patch = labels[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ]

        if patch.shape == (CHIP_SIZE, CHIP_SIZE):
            label_chips.append(patch)

    label_chips = np.array(label_chips, dtype=np.uint8)

    output_file = CHIP_DIR / f"label_chips_{year}.npy"

    np.save(output_file, label_chips)

    print("Saved:", output_file)
    print("Label chips shape:", label_chips.shape)

    # Class distribution
    print("\nClass distribution:")

    class_names = {
        0: "Urban",
        1: "Vegetation",
        2: "Water",
        3: "Bare Land",
        4: "Agriculture"
    }

    for class_id, class_name in class_names.items():

        percentage = (
            np.sum(label_chips == class_id)
            / label_chips.size
            * 100
        )

        print(
            f"Class {class_id} - "
            f"{class_name}: {percentage:.2f}%"
        )


if __name__ == "__main__":

    create_label_chips("2015")
    create_label_chips("2023")

    print("\nLabel chip generation completed!")