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