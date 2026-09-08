import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os

# ============================================================
# Exercise 4.3 - Candidate Grid Generation
# Study Area: Mumbai Municipal Corporation
# ============================================================

# Verified Mumbai Municipal Corporation bounding box
STUDY_AREA_BBOX = (
    72.776333,    # LON_MIN
    18.8939564,   # LAT_MIN
    72.9806446,   # LON_MAX
    19.2701767    # LAT_MAX
)

# Grid spacing
# 0.005 degree ≈ approximately 500 m
GRID_SPACING_DEGREES = 0.005


def create_grid():

    min_lon, min_lat, max_lon, max_lat = STUDY_AREA_BBOX

    # Generate longitude and latitude
    lons = np.arange(
        min_lon,
        max_lon,
        GRID_SPACING_DEGREES
    )

    lats = np.arange(
        min_lat,
        max_lat,
        GRID_SPACING_DEGREES
    )

    # Create candidate points
    points = [
        Point(lon, lat)
        for lon in lons
        for lat in lats
    ]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"geometry": points},
        crs="EPSG:4326"
    )

    # Candidate ID
    gdf["location_id"] = range(
        len(gdf)
    )

    # Longitude
    gdf["longitude"] = (
        gdf.geometry.x
    )

    # Latitude
    gdf["latitude"] = (
        gdf.geometry.y
    )

    print(
        f"Created {len(gdf)} candidate locations "
        f"({len(lons)} x {len(lats)})"
    )

    return gdf


if __name__ == "__main__":

    grid = create_grid()

    # Create output folder if required
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # Save candidate grid
    grid.to_file(
        "data/processed/candidate_locations.shp"
    )

    print(
        "Saved: data/processed/candidate_locations.shp"
    )