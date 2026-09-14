import numpy as np
from pathlib import Path


CHIP_DIR = Path("data/satellite/chips")


def fix_nan_chips(year):

    print(f"\n========== {year} ==========")

    input_file = CHIP_DIR / f"chips_{year}.npy"

    # Load image chips
    chips = np.load(input_file)

    print("Original shape:", chips.shape)
    print("Original NaN values:", np.isnan(chips).sum())

    # Replace NaN values band by band
    for band in range(chips.shape[3]):

        band_data = chips[:, :, :, band]

        # Calculate mean using valid values only
        band_mean = np.nanmean(band_data)

        # Find NaN locations
        nan_mask = np.isnan(band_data)

        # Replace NaN with band mean
        band_data[nan_mask] = band_mean

        chips[:, :, :, band] = band_data

        print(
            f"Band {band}: "
            f"NaN replaced with mean = {band_mean:.6f}"
        )

    # Replace any remaining invalid values
    chips = np.nan_to_num(
        chips,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    output_file = CHIP_DIR / f"chips_{year}_clean.npy"

    np.save(output_file, chips)

    print("Saved:", output_file)
    print("Final NaN values:", np.isnan(chips).sum())
    print("Final infinite values:", np.isinf(chips).sum())
    print("Minimum:", chips.min())
    print("Maximum:", chips.max())


if __name__ == "__main__":

    fix_nan_chips("2015")
    fix_nan_chips("2023")

    print("\nNaN fixing completed!")