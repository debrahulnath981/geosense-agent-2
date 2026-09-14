import numpy as np
from pathlib import Path


# ============================================================
# DIRECTORIES
# ============================================================

CLEAN_DIR = Path("data/satellite/clean")

CHIP_DIR = Path("data/satellite/chips")

CHIP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHIP SETTINGS
# ============================================================

CHIP_SIZE = 224

OVERLAP = 50

STRIDE = CHIP_SIZE - OVERLAP


# ============================================================
# CREATE CHIPS
# ============================================================

def create_chips(
    image: np.ndarray,
    year: str
):

    """
    Create 224 x 224 image chips.

    Input:
        image = (bands, height, width)

    Output:
        chips = (224, 224, bands)
    """

    bands, height, width = image.shape

    print(
        f"{year}: input shape = "
        f"{image.shape}"
    )

    print(
        f"Chip size = {CHIP_SIZE}"
    )

    print(
        f"Overlap = {OVERLAP}"
    )

    print(
        f"Stride = {STRIDE}"
    )

    chips = []

    positions = []

    # --------------------------------------------------------
    # SLIDING WINDOW
    # --------------------------------------------------------

    for row in range(
        0,
        height - CHIP_SIZE + 1,
        STRIDE
    ):

        for col in range(
            0,
            width - CHIP_SIZE + 1,
            STRIDE
        ):

            chip = image[
                :,
                row:row + CHIP_SIZE,
                col:col + CHIP_SIZE
            ]

            # ------------------------------------------------
            # CONVERT:
            # (bands, height, width)
            # TO:
            # (height, width, bands)
            # ------------------------------------------------

            chip = np.transpose(
                chip,
                (1, 2, 0)
            )

            chips.append(chip)

            positions.append(
                (row, col)
            )

    chips = np.asarray(
        chips,
        dtype=np.float32
    )

    positions = np.asarray(
        positions,
        dtype=np.int32
    )

    print(
        f"{year}: created "
        f"{len(chips)} chips"
    )

    print(
        f"{year}: chip array shape = "
        f"{chips.shape}"
    )

    return chips, positions


# ============================================================
# PROCESS ONE YEAR
# ============================================================

def process_year(year: str):

    input_file = (
        CLEAN_DIR /
        f"clean_{year}.npy"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found: {input_file}"
        )

    print()
    print("=" * 60)
    print(f"PROCESSING {year}")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = np.load(
        str(input_file)
    )

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if image.ndim != 3:

        raise ValueError(
            f"Expected 3D array, "
            f"got {image.ndim}D"
        )

    if image.shape[0] != 6:

        raise ValueError(
            f"Expected 6 bands, "
            f"got {image.shape[0]}"
        )

    # --------------------------------------------------------
    # CREATE CHIPS
    # --------------------------------------------------------

    chips, positions = create_chips(
        image,
        year
    )

    # --------------------------------------------------------
    # SAVE CHIPS
    # --------------------------------------------------------

    chip_file = (
        CHIP_DIR /
        f"chips_{year}.npy"
    )

    position_file = (
        CHIP_DIR /
        f"positions_{year}.npy"
    )

    np.save(
        str(chip_file),
        chips
    )

    np.save(
        str(position_file),
        positions
    )

    print()
    print(
        f"Saved chips: {chip_file}"
    )

    print(
        f"Saved positions: "
        f"{position_file}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print(
        f"Number of chips: "
        f"{len(chips)}"
    )

    print(
        f"Each chip: "
        f"{CHIP_SIZE} x {CHIP_SIZE}"
    )

    print(
        f"Bands per chip: "
        f"{chips.shape[-1]}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GeoSense Phase 2 — Image Chipping")
    print("=" * 60)

    print()
    print(
        f"Chip size : {CHIP_SIZE} x {CHIP_SIZE}"
    )

    print(
        f"Overlap   : {OVERLAP} pixels"
    )

    print(
        f"Stride    : {STRIDE} pixels"
    )

    # Process both years
    process_year("2015")

    process_year("2023")

    print()
    print("=" * 60)
    print("CHIPPING COMPLETE")
    print("=" * 60)

    print(
        f"Output directory: {CHIP_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()