import geopandas as gpd
import pandas as pd
import numpy as np
from db_connection import get_engine


# ============================================================
# SETTINGS
# ============================================================

# Mumbai / MCGM is in UTM Zone 43N
METRIC_CRS = 32643

CANDIDATE_FILE = "data/processed/candidate_locations.shp"
LANDUSE_FILE = "data/Shapefiles/osm_landuse.shp"

OUTPUT_FILE = "data/processed/features.csv"


# ============================================================
# LOAD POSTGIS LAYER
# ============================================================

def load_layer(sql, engine):
    return gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry"
    )


# ============================================================
# DISTANCE TO NEAREST FEATURE
# ============================================================

def calc_distance_to_nearest(candidates_gdf, layer_gdf):

    candidates_proj = candidates_gdf.to_crs(epsg=METRIC_CRS)
    layer_proj = layer_gdf.to_crs(epsg=METRIC_CRS)

    distances = []

    for point in candidates_proj.geometry:

        distance = layer_proj.geometry.distance(point).min()

        distances.append(distance)

    return pd.Series(distances, index=candidates_gdf.index)


# ============================================================
# LOAD HOSPITALS
# ============================================================

def load_hospitals(engine):

    sql = """
    SELECT geometry
    FROM osm_poi
    WHERE fclass = 'hospital'
    """

    hospitals = load_layer(sql, engine)

    print(f"Hospitals found: {len(hospitals)}")

    return hospitals


# ============================================================
# FLOOD RISK
# ============================================================

def assign_flood_risk(candidates_gdf, engine):

    try:

        flood = load_layer(
            """
            SELECT geometry, risk_level
            FROM flood_zones
            """,
            engine
        )

        joined = gpd.sjoin(
            candidates_gdf,
            flood,
            how="left",
            predicate="within"
        )

        risk_map = {
            "high": 8,
            "medium": 5,
            "low": 2
        }

        risk = (
            joined["risk_level"]
            .map(risk_map)
            .fillna(0)
        )

        return risk.reset_index(drop=True)

    except Exception:

        print(
            "WARNING: flood_zones table not found."
        )

        print(
            "flood_risk will be set to 0 as a placeholder."
        )

        return pd.Series(
            np.zeros(len(candidates_gdf))
        )


# ============================================================
# LOAD LAND USE
# ============================================================

def load_landuse():

    landuse = gpd.read_file(LANDUSE_FILE)

    print(f"Land-use features found: {len(landuse)}")

    print("\nLand-use classes:")

    print(
        landuse["fclass"]
        .value_counts()
        .to_string()
    )

    return landuse


# ============================================================
# NEAREST LAND-USE CLASS
# ============================================================

def nearest_landuse_class(candidates_gdf, landuse):

    candidates_proj = candidates_gdf.to_crs(
        epsg=METRIC_CRS
    )

    landuse_proj = landuse.to_crs(
        epsg=METRIC_CRS
    )

    classes = []

    for point in candidates_proj.geometry:

        distances = landuse_proj.geometry.distance(point)

        nearest_index = distances.idxmin()

        classes.append(
            landuse_proj.loc[
                nearest_index,
                "fclass"
            ]
        )

    return pd.Series(
        classes,
        index=candidates_gdf.index
    )


# ============================================================
# DISTANCE TO COMMERCIAL LAND USE
# ============================================================

def distance_to_commercial(candidates_gdf, landuse):

    commercial = landuse[
        landuse["fclass"].str.lower() == "commercial"
    ].copy()

    if len(commercial) == 0:

        print(
            "\nWARNING: No 'commercial' land-use class found."
        )

        print(
            "dist_landuse_commercial_m will be NaN."
        )

        return pd.Series(
            np.nan,
            index=candidates_gdf.index
        )

    print(
        f"Commercial land-use features found: {len(commercial)}"
    )

    return calc_distance_to_nearest(
        candidates_gdf,
        commercial
    )


# ============================================================
# BUILD FEATURE TABLE
# ============================================================

def build_feature_table(candidates_gdf, engine):

    print("\n==============================================")
    print("STARTING FEATURE ENGINEERING")
    print("==============================================")

    # --------------------------------------------------------
    # Candidate ID
    # --------------------------------------------------------

    if "location_id" in candidates_gdf.columns:

        location_id = candidates_gdf["location_id"]

    elif "location_i" in candidates_gdf.columns:

        print(
            "Shapefile field 'location_i' detected."
        )

        location_id = candidates_gdf["location_i"]

    else:

        print(
            "No location ID field found."
        )

        print(
            "Creating location IDs from row numbers."
        )

        location_id = range(len(candidates_gdf))

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    df = pd.DataFrame()

    df["location_id"] = list(location_id)

    df["latitude"] = candidates_gdf.geometry.y

    df["longitude"] = candidates_gdf.geometry.x

    # --------------------------------------------------------
    # ROAD DISTANCE
    # --------------------------------------------------------

    print("\nCalculating distance to nearest road...")

    roads = load_layer(
        "SELECT geometry FROM osm_roads",
        engine
    )

    print(
        f"Road features found: {len(roads)}"
    )

    df["dist_road_m"] = calc_distance_to_nearest(
        candidates_gdf,
        roads
    )

    # --------------------------------------------------------
    # HOSPITAL DISTANCE
    # --------------------------------------------------------

    print("\nCalculating distance to nearest hospital...")

    hospitals = load_hospitals(engine)

    df["dist_hospital_m"] = calc_distance_to_nearest(
        candidates_gdf,
        hospitals
    )

    # --------------------------------------------------------
    # SCHOOL DISTANCE
    # --------------------------------------------------------

    print("\nCalculating distance to nearest school...")

    try:

        schools = load_layer(
            "SELECT geometry FROM osm_poi_school",
            engine
        )

        print(
            f"School features found: {len(schools)}"
        )

        df["dist_school_m"] = calc_distance_to_nearest(
            candidates_gdf,
            schools
        )

    except Exception as e:

        print(
            "School table could not be loaded."
        )

        df["dist_school_m"] = np.nan

    # --------------------------------------------------------
    # POLICE DISTANCE
    # --------------------------------------------------------

    print("\nCalculating distance to nearest police station...")

    try:

        police = load_layer(
            "SELECT geometry FROM osm_poi_police",
            engine
        )

        print(
            f"Police features found: {len(police)}"
        )

        df["dist_police_m"] = calc_distance_to_nearest(
            candidates_gdf,
            police
        )

    except Exception:

        print(
            "Police table could not be loaded."
        )

        df["dist_police_m"] = np.nan

    # --------------------------------------------------------
    # MARKET DISTANCE
    # --------------------------------------------------------

    print("\nCalculating distance to nearest market...")

    try:

        markets = load_layer(
            "SELECT geometry FROM osm_poi_market",
            engine
        )

        print(
            f"Market features found: {len(markets)}"
        )

        df["dist_market_m"] = calc_distance_to_nearest(
            candidates_gdf,
            markets
        )

    except Exception:

        print(
            "Market table could not be loaded."
        )

        df["dist_market_m"] = np.nan

    # --------------------------------------------------------
    # FLOOD RISK
    # --------------------------------------------------------

    print("\nAssigning flood risk...")

    df["flood_risk"] = assign_flood_risk(
        candidates_gdf,
        engine
    ).values

    # --------------------------------------------------------
    # LAND USE
    # --------------------------------------------------------

    print("\nProcessing land-use data...")

    landuse = load_landuse()

    df["land_use_class"] = nearest_landuse_class(
        candidates_gdf,
        landuse
    ).values

    # --------------------------------------------------------
    # DISTANCE TO COMMERCIAL LAND USE
    # --------------------------------------------------------

    print(
        "\nCalculating distance to commercial land use..."
    )

    df["dist_landuse_commercial_m"] = (
        distance_to_commercial(
            candidates_gdf,
            landuse
        ).values
    )

    # --------------------------------------------------------
    # POPULATION / ELEVATION
    # --------------------------------------------------------

    # These datasets are not currently available
    # in the project, so they are left as NaN.

    df["population_density"] = np.nan

    df["elevation_m"] = np.nan

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("GeoSense - Exercise 4.4")
    print("Feature Engineering")
    print("Mumbai / MCGM")
    print("==============================================")

    print(
        f"\nMetric CRS: EPSG:{METRIC_CRS}"
    )

    # --------------------------------------------------------
    # Database connection
    # --------------------------------------------------------

    engine = get_engine()

    print(
        "\nDatabase connection successful."
    )

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    print(
        "\nLoading candidate locations..."
    )

    candidates = gpd.read_file(
        CANDIDATE_FILE
    )

    print(
        f"Candidate locations: {len(candidates)}"
    )

    print(
        "Candidate CRS:",
        candidates.crs
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = build_feature_table(
        candidates,
        engine
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n==============================================")
    print("FEATURE ENGINEERING COMPLETE")
    print("==============================================")

    print(
        f"\nOutput saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nRows: {len(features)}"
    )

    print(
        f"Columns: {len(features.columns)}"
    )

    print("\nFeature columns:")

    print(
        features.columns.tolist()
    )

    print("\nFirst 5 rows:")

    print(
        features.head().to_string(index=False)
    )