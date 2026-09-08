import os
import geopandas as gpd
from db_connection import get_engine


# ============================================================
# GeoSense Agent 2.0
# Exercise 4.2 - PostGIS Data Ingestion
# ============================================================

DATA_DIR = "data/Shapefiles"


# ------------------------------------------------------------
# Input shapefiles and PostGIS table names
# ------------------------------------------------------------

LAYERS_TO_LOAD = [
    ("osm_roads.shp", "osm_roads"),
    ("osm_Building.shp", "osm_buildings"),
    ("osm_landuse.shp", "osm_landuse"),
    ("osm_pois.shp", "osm_poi"),
]


# ------------------------------------------------------------
# Load one shapefile
# ------------------------------------------------------------

def load_layer(filename, table_name, engine):

    filepath = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 60)
    print(f"Loading: {filename}")
    print(f"PostGIS table: {table_name}")
    print("=" * 60)

    # Check file
    if not os.path.exists(filepath):
        print(f"ERROR: File not found:")
        print(filepath)
        return False

    # Read shapefile
    gdf = gpd.read_file(filepath)

    print(f"Features found: {len(gdf)}")
    print(f"Original CRS: {gdf.crs}")
    print(f"Geometry type: {gdf.geometry.geom_type.unique()}")

    # Check CRS
    if gdf.crs is None:
        print("ERROR: CRS is missing!")
        return False

    # Convert to WGS84
    gdf = gdf.to_crs(epsg=4326)

    print(f"Converted CRS: {gdf.crs}")

    # Load into PostGIS
    gdf.to_postgis(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(f"SUCCESS: Loaded {len(gdf)} features")
    print(f"PostGIS table: {table_name}")

    return True


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("GeoSense Agent 2.0")
    print("Exercise 4.2 - PostGIS Data Ingestion")
    print("=" * 60)

    # Connect to database
    print("\nConnecting to PostGIS...")

    engine = get_engine()

    print("Database connection successful!")

    # Load all layers
    success_count = 0

    for filename, table_name in LAYERS_TO_LOAD:

        success = load_layer(
            filename,
            table_name,
            engine
        )

        if success:
            success_count += 1

    # Final result
    print("\n" + "=" * 60)
    print("POSTGIS DATA INGESTION COMPLETED")
    print("=" * 60)

    print(f"\nLayers successfully loaded: "
          f"{success_count}/{len(LAYERS_TO_LOAD)}")

    print("\nExpected PostGIS tables:")

    for _, table_name in LAYERS_TO_LOAD:
        print(f"  - {table_name}")

    print("\nExercise 4.2 completed!")


if __name__ == "__main__":
    main()