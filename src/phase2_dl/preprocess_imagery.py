import numpy as np
import rasterio
from pathlib import Path
import json


# ============================================================
# DIRECTORIES
# ============================================================

SAT_DIR = Path("data/satellite")

RAW_DIR = SAT_DIR / "raw"

CLEAN_DIR = SAT_DIR / "clean"

CLEAN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BAND INFORMATION
# ============================================================

BAND_NAMES = [
    "B2_Blue",
    "B3_Green",
    "B4_Red",
    "B8_NIR",
    "B11_SWIR1",
    "B12_SWIR2"
]


BAND_INDICES = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
    "B11": 4,
    "B12": 5
}


# ============================================================
# REMOVE CLOUDS
# ============================================================

def remove_clouds(
    image_array: np.ndarray,
    cloud_threshold: float = 3000
) -> np.ndarray:

    """
    Remove very bright pixels interpreted as clouds.

    A pixel is considered cloudy when ALL six
    Sentinel-2 reflectance bands are greater than 3000.

    Cloud pixels are replaced with NaN.
    """

    # Cloud mask
    cloud_mask = np.all(
        image_array > cloud_threshold,
        axis=0
    )

    # Convert to floating point
    clean = image_array.astype(float)

    # Replace cloud pixels with NaN
    clean[:, cloud_mask] = np.nan

    cloud_pct = (
        cloud_mask.mean() * 100
    )

    print(
        f" Cloud coverage removed: "
        f"{cloud_pct:.1f}%"
    )

    return clean


# ============================================================
# NORMALISE TO 0–1
# ============================================================

def normalise_to_01(
    image_array: np.ndarray
) -> np.ndarray:

    """
    Convert Sentinel-2 reflectance values
    from 0–10000 to 0–1.
    """

    normalised = np.clip(
        image_array / 10000.0,
        0,
        1
    )

    return normalised


# ============================================================
# NDVI
# ============================================================

def compute_ndvi(
    image_array: np.ndarray
) -> np.ndarray:

    """
    NDVI = (NIR - Red) / (NIR + Red)

    B8 = NIR
    B4 = Red
    """

    nir = image_array[
        BAND_INDICES["B8"]
    ].astype(float)

    red = image_array[
        BAND_INDICES["B4"]
    ].astype(float)

    denominator = nir + red

    denominator[
        denominator == 0
    ] = np.nan

    ndvi = (
        (nir - red)
        / denominator
    )

    return np.clip(
        ndvi,
        -1,
        1
    )


# ============================================================
# NDWI
# ============================================================

def compute_ndwi(
    image_array: np.ndarray
) -> np.ndarray:

    """
    NDWI = (Green - NIR) / (Green + NIR)

    B3 = Green
    B8 = NIR
    """

    green = image_array[
        BAND_INDICES["B3"]
    ].astype(float)

    nir = image_array[
        BAND_INDICES["B8"]
    ].astype(float)

    denominator = green + nir

    denominator[
        denominator == 0
    ] = np.nan

    ndwi = (
        (green - nir)
        / denominator
    )

    return np.clip(
        ndwi,
        -1,
        1
    )


# ============================================================
# PROCESS ONE TIFF
# ============================================================

def process_file(
    tif_path: Path
) -> dict:

    print()
    print("=" * 60)
    print(f"Processing: {tif_path.name}")
    print("=" * 60)

    # --------------------------------------------------------
    # READ RASTER
    # --------------------------------------------------------

    with rasterio.open(
        str(tif_path)
    ) as src:

        image = src.read()

        meta = src.meta.copy()

        transform = src.transform

        print(
            f"Shape: {image.shape}"
        )

        print(
            f"Dtype: {image.dtype}"
        )

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Resolution: "
            f"{src.res}"
        )

    # --------------------------------------------------------
    # CHECK BANDS
    # --------------------------------------------------------

    if image.shape[0] != 6:

        raise ValueError(
            f"Expected 6 bands, "
            f"but found {image.shape[0]}"
        )

    # --------------------------------------------------------
    # STEP 1 — REMOVE CLOUDS
    # --------------------------------------------------------

    clean = remove_clouds(
        image
    )

    # --------------------------------------------------------
    # STEP 2 — NORMALISE
    # --------------------------------------------------------

    normalised = normalise_to_01(
        clean
    )

    # --------------------------------------------------------
    # STEP 3 — NDVI
    # --------------------------------------------------------

    ndvi = compute_ndvi(
        clean
    )

    # --------------------------------------------------------
    # STEP 4 — NDWI
    # --------------------------------------------------------

    ndwi = compute_ndwi(
        clean
    )

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    year = tif_path.stem.split("_")[-1]

    # --------------------------------------------------------
    # SAVE NORMALISED IMAGE
    # --------------------------------------------------------

    clean_path = (
        CLEAN_DIR
        / f"clean_{year}.npy"
    )

    np.save(
        str(clean_path),
        normalised
    )

    # --------------------------------------------------------
    # SAVE NDVI
    # --------------------------------------------------------

    ndvi_path = (
        CLEAN_DIR
        / f"ndvi_{year}.npy"
    )

    np.save(
        str(ndvi_path),
        ndvi
    )

    # --------------------------------------------------------
    # SAVE NDWI
    # --------------------------------------------------------

    ndwi_path = (
        CLEAN_DIR
        / f"ndwi_{year}.npy"
    )

    np.save(
        str(ndwi_path),
        ndwi
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    cloud_free_pct = (
        np.mean(
            ~np.isnan(
                normalised[0]
            )
        )
        * 100
    )

    stats = {

        "year": year,

        "shape": list(
            image.shape
        ),

        "ndvi_mean": float(
            np.nanmean(ndvi)
        ),

        "ndvi_std": float(
            np.nanstd(ndvi)
        ),

        "ndwi_mean": float(
            np.nanmean(ndwi)
        ),

        "cloud_free_pct": float(
            cloud_free_pct
        )
    }

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    meta_path = (
        CLEAN_DIR
        / f"meta_{year}.json"
    )

    with open(
        str(meta_path),
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()

    print(
        f"NDVI mean: "
        f"{stats['ndvi_mean']:.3f}"
    )

    print(
        f"NDWI mean: "
        f"{stats['ndwi_mean']:.3f}"
    )

    print(
        f"Cloud-free: "
        f"{stats['cloud_free_pct']:.1f}%"
    )

    print()

    print(
        f"Saved: {clean_path}"
    )

    print(
        f"Saved: {ndvi_path}"
    )

    print(
        f"Saved: {ndwi_path}"
    )

    print(
        f"Saved: {meta_path}"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "GeoSense Phase 2 — "
        "Sentinel-2 Preprocessing"
    )
    print("=" * 60)

    # Find only Mumbai files
    tif_files = sorted(
        RAW_DIR.glob(
            "sentinel2_Mumbai_*.tif"
        )
    )

    if not tif_files:

        print(
            "No Mumbai Sentinel-2 "
            "GeoTIFF files found."
        )

        print(
            f"Expected location: {RAW_DIR}"
        )

        return

    print()
    print(
        f"Found {len(tif_files)} "
        "Mumbai files:"
    )

    for file in tif_files:

        print(
            f"  - {file.name}"
        )

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    results = []

    for tif_file in tif_files:

        stats = process_file(
            tif_file
        )

        results.append(
            stats
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for stats in results:

        print(
            f"{stats['year']} | "
            f"NDVI mean = "
            f"{stats['ndvi_mean']:.3f} | "
            f"NDWI mean = "
            f"{stats['ndwi_mean']:.3f} | "
            f"Cloud-free = "
            f"{stats['cloud_free_pct']:.1f}%"
        )

    print()
    print(
        "Preprocessing complete!"
    )

    print(
        f"Output directory: {CLEAN_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()