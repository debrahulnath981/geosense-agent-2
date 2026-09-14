import numpy as np
from pathlib import Path

# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

CLEAN_DIR = Path("data/satellite/clean")
LABEL_DIR = Path("data/satellite/labels")

LABEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LAND COVER CLASSES
# --------------------------------------------------

CLASSES = {
    0: "Urban",
    1: "Vegetation",
    2: "Water",
    3: "Bare Land",
    4: "Agriculture"
}


# --------------------------------------------------
# PIXEL CLASSIFICATION
# --------------------------------------------------

def classify_pixel(ndvi, ndwi, nir, swir):

    # Water
    if ndwi > 0.3:
        return 2

    # Vegetation
    if ndvi > 0.4:
        return 1

    # Agriculture
    if ndvi > 0.15:
        return 4

    # Urban
    if ndvi < 0.05 and swir > 0.2:
        return 0

    # Bare Land
    return 3


# --------------------------------------------------
# CREATE LABEL MASK
# --------------------------------------------------

def create_label_mask(year):

    print(f"\nGenerating labels for {year}...")

    ndvi = np.load(CLEAN_DIR / f"ndvi_{year}.npy")
    ndwi = np.load(CLEAN_DIR / f"ndwi_{year}.npy")
    clean = np.load(CLEAN_DIR / f"clean_{year}.npy")

    rows, cols = ndvi.shape

    label_mask = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):

            nir = float(clean[3, r, c])
            swir = float(clean[4, r, c])

            label_mask[r, c] = classify_pixel(
                ndvi[r, c],
                ndwi[r, c],
                nir,
                swir
            )

    # Save label mask
    label_path = LABEL_DIR / f"labels_{year}.npy"

    np.save(label_path, label_mask)

    print(f"Saved: {label_path}")
    print(f"Shape: {label_mask.shape}")

    # Class distribution
    print("\nClass distribution:")

    for class_id, class_name in CLASSES.items():

        percentage = (
            np.sum(label_mask == class_id)
            / label_mask.size
            * 100
        )

        print(
            f"Class {class_id} - "
            f"{class_name}: {percentage:.2f}%"
        )

    return label_mask


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    create_label_mask("2015")
    create_label_mask("2023")

    print("\nLabel generation completed!")