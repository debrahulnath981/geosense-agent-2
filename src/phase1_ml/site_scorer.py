# ============================================================
# site_scorer.py
# GeoSense Agent 2.0 - Practical Exercise 5.3
# ============================================================

import os
import joblib
import numpy as np
import geopandas as gpd
import shap
from shapely.geometry import Point
from db_connection import get_engine


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/saved/site_scorer_model.pkl"

# Mumbai uses UTM Zone 43N
METRIC_CRS = 32643

FEATURE_NAMES = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk"
]


# ============================================================
# LOAD MODEL
# ============================================================

_model = None
_explainer = None


def _load_model():
    """
    Load the trained model and SHAP explainer.
    The model is loaded only once.
    """

    global _model, _explainer

    if _model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run train_model.py first."
            )

        print("Loading trained model...")

        _model = joblib.load(MODEL_PATH)

        _explainer = shap.TreeExplainer(_model)

        print("Model loaded successfully!")

    return _model, _explainer


# ============================================================
# CALCULATE FEATURES FOR ONE LOCATION
# ============================================================

def _get_features_for_point(lat, lon, engine):
    """
    Calculate geographic features for a single latitude/longitude.
    """

    # Create point in WGS84
    point = Point(lon, lat)

    point_gdf = gpd.GeoDataFrame(
        [{"geometry": point}],
        crs="EPSG:4326"
    )

    # Reproject to metre-based CRS
    point_projected = point_gdf.to_crs(
        epsg=METRIC_CRS
    )

    point_geom = point_projected.geometry.iloc[0]


    # --------------------------------------------------------
    # Distance to nearest feature
    # --------------------------------------------------------

    def distance_to_layer(table_name, where_clause=None):

        if where_clause:

            query = (
                f"SELECT geometry FROM {table_name} "
                f"WHERE {where_clause}"
            )

        else:

            query = (
                f"SELECT geometry FROM {table_name}"
            )

        layer = gpd.read_postgis(
            query,
            engine,
            geom_col="geometry"
        )

        if layer.empty:
            raise ValueError(
                f"No features found in {table_name}"
            )

        layer = layer.to_crs(
            epsg=METRIC_CRS
        )

        distance = layer.geometry.distance(
            point_geom
        ).min()

        return float(distance)


    # --------------------------------------------------------
    # Road distance
    # --------------------------------------------------------

    dist_road = distance_to_layer(
        "osm_roads"
    )


    # --------------------------------------------------------
    # Hospital distance
    # --------------------------------------------------------

    dist_hospital = distance_to_layer(
        "osm_poi",
        "fclass = 'hospital'"
    )


    # --------------------------------------------------------
    # Flood risk
    # --------------------------------------------------------
    # Flood-zone dataset is currently unavailable,
    # therefore the same placeholder used during
    # feature engineering is applied.

    flood_risk = 0.0


    # --------------------------------------------------------
    # Create feature dictionary
    # --------------------------------------------------------

    features = {

        "dist_road_m": dist_road,

        "dist_hospital_m": dist_hospital,

        "flood_risk": flood_risk
    }

    return features


# ============================================================
# MAIN SITE SCORING FUNCTION
# ============================================================

def score_location(lat: float, lon: float) -> dict:
    """
    Score a geographic location for site suitability.

    Parameters
    ----------
    lat : float
        Latitude of the location.

    lon : float
        Longitude of the location.

    Returns
    -------
    dict
        Contains:
        - latitude
        - longitude
        - opportunity_score
        - risk_score
        - features
        - shap_explanation
        - verdict
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, explainer = _load_model()


    # --------------------------------------------------------
    # Connect to PostGIS
    # --------------------------------------------------------

    engine = get_engine()


    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    features = _get_features_for_point(
        lat,
        lon,
        engine
    )


    # --------------------------------------------------------
    # Convert features to model input
    # --------------------------------------------------------

    X = np.array([
        [
            features["dist_road_m"],
            features["dist_hospital_m"],
            features["flood_risk"]
        ]
    ])


    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[0]

    poor_probability = float(probabilities[0])

    good_probability = float(probabilities[1])


    # --------------------------------------------------------
    # Convert probability to 0-10 scores
    # --------------------------------------------------------

    opportunity_score = round(
        good_probability * 10,
        2
    )

    risk_score = round(
        poor_probability * 10,
        2
    )


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = explainer.shap_values(X)


    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        shap_positive = shap_values[1][0]

    elif isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:

            shap_positive = shap_values[0, :, 1]

        else:

            shap_positive = shap_values[0]

    else:

        shap_positive = shap_values


    # --------------------------------------------------------
    # Create SHAP explanation dictionary
    # --------------------------------------------------------

    shap_explanation = {

        name: round(
            float(value),
            4
        )

        for name, value in zip(
            FEATURE_NAMES,
            shap_positive
        )
    }


    # --------------------------------------------------------
    # Determine verdict
    # --------------------------------------------------------

    if opportunity_score > 5:

        verdict = "GOOD SITE"

    else:

        verdict = "POOR SITE"


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        "latitude": lat,

        "longitude": lon,

        "opportunity_score": opportunity_score,

        "risk_score": risk_score,

        "features": features,

        "shap_explanation": shap_explanation,

        "verdict": verdict
    }


# ============================================================
# TEST THE MODULE
# ============================================================

# ============================================================
# TEST BOTH GOOD AND POOR SITES
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("       GeoSense Site Scorer Test")
    print("       Mumbai Municipal Corporation")
    print("==============================================")


    # --------------------------------------------------------
    # TEST LOCATION 1 - GOOD SITE
    # --------------------------------------------------------

    good_lat = 19.0760
    good_lon = 72.8777

    print("\n\n==============================================")
    print("TEST 1: GOOD SITE")
    print("==============================================")

    good_result = score_location(
        good_lat,
        good_lon
    )

    print("\nRESULT")
    print("----------------------------------------------")

    print(f"Latitude          : {good_result['latitude']}")
    print(f"Longitude         : {good_result['longitude']}")
    print(f"Opportunity Score : {good_result['opportunity_score']}/10")
    print(f"Risk Score        : {good_result['risk_score']}/10")
    print(f"Verdict            : {good_result['verdict']}")

    print("\nFeatures:")
    for name, value in good_result["features"].items():
        print(f"  {name:20s}: {value}")

    print("\nSHAP Explanation:")
    for name, value in good_result["shap_explanation"].items():
        print(f"  {name:20s}: {value}")


    # --------------------------------------------------------
    # TEST LOCATION 2 - POOR SITE
    # --------------------------------------------------------

    poor_lat = 18.893956
    poor_lon = 72.776333

    print("\n\n==============================================")
    print("TEST 2: POOR SITE")
    print("==============================================")

    poor_result = score_location(
        poor_lat,
        poor_lon
    )

    print("\nRESULT")
    print("----------------------------------------------")

    print(f"Latitude          : {poor_result['latitude']}")
    print(f"Longitude         : {poor_result['longitude']}")
    print(f"Opportunity Score : {poor_result['opportunity_score']}/10")
    print(f"Risk Score        : {poor_result['risk_score']}/10")
    print(f"Verdict            : {poor_result['verdict']}")

    print("\nFeatures:")
    for name, value in poor_result["features"].items():
        print(f"  {name:20s}: {value}")

    print("\nSHAP Explanation:")
    for name, value in poor_result["shap_explanation"].items():
        print(f"  {name:20s}: {value}")


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n==============================================")
    print("              FINAL SUMMARY")
    print("==============================================")

    print(
        f"GOOD SITE : "
        f"Opportunity={good_result['opportunity_score']}/10, "
        f"Risk={good_result['risk_score']}/10, "
        f"Verdict={good_result['verdict']}"
    )

    print(
        f"POOR SITE : "
        f"Opportunity={poor_result['opportunity_score']}/10, "
        f"Risk={poor_result['risk_score']}/10, "
        f"Verdict={poor_result['verdict']}"
    )

    print("\n==============================================")
    print("Both site scoring tests completed successfully!")
    print("==============================================")

    # ============================================================
# test_site_scorer.py
# GeoSense Agent 2.0 - Practical Exercise 5.4
# ============================================================

import sys

# Allow Python to find site_scorer.py
sys.path.append("src/phase1_ml")

from site_scorer import score_location


# ------------------------------------------------------------
# Test 1: Result should be a dictionary
# ------------------------------------------------------------

def test_score_returns_dict():

    result = score_location(
        19.0760,
        72.8777
    )

    assert isinstance(
        result,
        dict
    )


# ------------------------------------------------------------
# Test 2: Required keys should exist
# ------------------------------------------------------------

def test_score_has_required_keys():

    result = score_location(
        19.0760,
        72.8777
    )

    required_keys = [
        "latitude",
        "longitude",
        "opportunity_score",
        "risk_score",
        "features",
        "shap_explanation",
        "verdict"
    ]

    for key in required_keys:

        assert key in result


# ------------------------------------------------------------
# Test 3: Scores must be between 0 and 10
# ------------------------------------------------------------

def test_scores_in_valid_range():

    result = score_location(
        19.0760,
        72.8777
    )

    assert 0 <= result["opportunity_score"] <= 10

    assert 0 <= result["risk_score"] <= 10


# ------------------------------------------------------------
# Test 4: Verdict must be valid
# ------------------------------------------------------------

def test_verdict_is_valid():

    result = score_location(
        19.0760,
        72.8777
    )

    assert result["verdict"] in [
        "GOOD SITE",
        "POOR SITE"
    ]