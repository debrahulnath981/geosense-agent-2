import ee
import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# LOAD .env
# ============================================================

load_dotenv()

# ============================================================
# STUDY AREA — MUMBAI
# ============================================================

BBOX = [
    float(os.getenv("BBOX_MIN_LON", "72.776333")),
    float(os.getenv("BBOX_MIN_LAT", "18.8939564")),
    float(os.getenv("BBOX_MAX_LON", "72.9806446")),
    float(os.getenv("BBOX_MAX_LAT", "19.2701767"))
]

STUDY_NAME = os.getenv("STUDY_AREA", "Mumbai")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUT_DIR = Path("data/satellite/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EARTH ENGINE PROJECT
# ============================================================

EE_PROJECT = "core-guard-499011-e8"


# ============================================================
# STUDY AREA
# ============================================================

def get_study_area():
    """Return Mumbai bounding box as an Earth Engine geometry."""
    return ee.Geometry.Rectangle(BBOX)


# ============================================================
# SENTINEL-2 IMAGE COLLECTION
# ============================================================

def get_clean_image(year: int) -> ee.Image:
    """
    Create a median Sentinel-2 composite.

    Dataset:
        COPERNICUS/S2_SR_HARMONIZED

    Cloud cover:
        < 10%

    Bands:
        B2, B3, B4, B8, B11, B12, QA60
    """

    region = get_study_area()

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"Searching Sentinel-2 images for {year}...")

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                10
            )
        )
        .select([
            "B2",
            "B3",
            "B4",
            "B8",
            "B11",
            "B12",
            "QA60"
        ])
    )

    count = collection.size().getInfo()

    print(f"Year {year}: found {count} clean images")

    if count == 0:
        raise ValueError(
            f"No clean Sentinel-2 images found for {year}."
        )

    # Median composite
    median_image = (
        collection
        .median()
        .clip(region)
    )

    return median_image


# ============================================================
# GOOGLE DRIVE EXPORT
# ============================================================

def export_to_drive(
    image: ee.Image,
    year: int,
    bands: list
):
    """
    Export Sentinel-2 composite to Google Drive.
    """

    description = f"Sentinel2_{STUDY_NAME}_{year}"

    file_name = f"sentinel2_{STUDY_NAME}_{year}"

    region = get_study_area().bounds()

    print()
    print(f"Creating Google Drive export for {year}...")
    print(f"Description: {description}")
    print(f"File name: {file_name}")
    print("Scale: 10 metres")

    task = ee.batch.Export.image.toDrive(
        image=image.select(bands),
        description=description,
        folder="GeoSense_Satellite",
        fileNamePrefix=file_name,
        region=region.getInfo()["coordinates"],
        scale=10,
        fileFormat="GeoTIFF",
        maxPixels=1e13
    )

    task.start()

    print(
        f"Export task started successfully for {year}"
    )

    print(f"Task ID: {task.id}")

    return task


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GeoSense Phase 2 — Sentinel-2 Data Acquisition")
    print("=" * 60)

    # --------------------------------------------------------
    # INITIALIZE EARTH ENGINE
    # --------------------------------------------------------

    print()
    print("Initialising Google Earth Engine...")

    ee.Initialize(
        project=EE_PROJECT
    )

    print(
        "Earth Engine initialized successfully!"
    )

    # --------------------------------------------------------
    # STUDY INFORMATION
    # --------------------------------------------------------

    print()
    print(f"Study area : {STUDY_NAME}")
    print(f"BBOX       : {BBOX}")
    print(f"GEE Project: {EE_PROJECT}")

    # --------------------------------------------------------
    # YEARS
    # --------------------------------------------------------

    years_to_download = [
        2015,
        2023
    ]

    # --------------------------------------------------------
    # SIX INPUT BANDS
    # --------------------------------------------------------

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12"
    ]

    print()
    print("Bands to export:")
    print(bands)

    # --------------------------------------------------------
    # EXPORT BOTH YEARS
    # --------------------------------------------------------

    tasks = []

    for year in years_to_download:

        print()
        print("-" * 60)
        print(f"PROCESSING YEAR: {year}")
        print("-" * 60)

        try:

            image = get_clean_image(year)

            task = export_to_drive(
                image,
                year,
                bands
            )

            tasks.append(
                (
                    year,
                    task.id
                )
            )

        except Exception as e:

            print()
            print(f"ERROR for {year}:")
            print(e)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    for year, task_id in tasks:

        print(
            f"Year {year}: Export task started"
        )

        print(
            f"Task ID: {task_id}"
        )

    print()
    print("Google Drive folder:")
    print("GeoSense_Satellite")

    print()
    print(
        "Earth Engine exports run in the background."
    )

    print(
        "Wait until both tasks show COMPLETED."
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()