# ============================================================
# GeoSense Agent 2.0 — Phase 2 Core Image Classifier
# Exercise 5
#
# INPUT:
#   latitude, longitude, radius_m
#
# OUTPUT:
#   land_cover
#   class_id
#   confidence_pct
#   ndvi
#   ndwi
#   ndvi_label
#   class_distribution
#   change_flag
#   change_description
# ============================================================

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "saved"
CLEAN_DIR = ROOT / "data" / "satellite" / "clean"

PRITHVI_PATH = MODEL_DIR / "prithvi_finetuned.pth"
UNET_PATH = MODEL_DIR / "unet_best.pth"

IMAGE_2023_PATH = CLEAN_DIR / "clean_2023.npy"
IMAGE_2015_PATH = CLEAN_DIR / "clean_2015.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. MUMBAI STUDY AREA BBOX
# ============================================================

BBOX_MIN_LON = float(os.getenv("BBOX_MIN_LON", "72.776333"))
BBOX_MIN_LAT = float(os.getenv("BBOX_MIN_LAT", "18.8939564"))
BBOX_MAX_LON = float(os.getenv("BBOX_MAX_LON", "72.9806446"))
BBOX_MAX_LAT = float(os.getenv("BBOX_MAX_LAT", "19.2701767"))

STUDY_AREA = os.getenv("STUDY_AREA", "Mumbai")


# ============================================================
# 3. LAND COVER CLASSES
# ============================================================

CLASSES = {
    0: "Urban / Built-up",
    1: "Dense Vegetation / Forest",
    2: "Water Body",
    3: "Bare Land / Soil",
    4: "Agriculture / Crops",
}


# ============================================================
# 4. PRITHVI NORMALIZATION VALUES
# ============================================================
# Official Prithvi-100M configuration used during fine-tuning.
# Six input bands are B02, B03, B04, B05/B06/B07 slots
# adapted to the six Sentinel-2 bands used in this project.

PRITHVI_MEAN = torch.tensor(
    [
        775.229,
        1080.992,
        1228.585,
        2497.202,
        2204.213,
        1610.832,
    ],
    dtype=torch.float32,
).view(6, 1, 1)

PRITHVI_STD = torch.tensor(
    [
        1281.526,
        1270.029,
        1399.480,
        1368.344,
        1291.676,
        1154.505,
    ],
    dtype=torch.float32,
).view(6, 1, 1)


# ============================================================
# 5. GLOBAL CACHES
# ============================================================

_model = None
_model_name = None

_image_2023 = None
_image_2015 = None


# ============================================================
# 6. PRITHVI SEGMENTATION MODEL
# ============================================================

try:
    from .prithvi_mae import PrithviMAE
except ImportError:
    from prithvi_mae import PrithviMAE


class PrithviSegmentation(nn.Module):
    """
    Prithvi-100M backbone + lightweight segmentation decoder.

    Input:
        [B, 6, 3, 224, 224]

    Output:
        [B, 5, 224, 224]
    """

    def __init__(self, prithvi_model, num_classes=5):
        super().__init__()

        self.prithvi = prithvi_model

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):

        features = self.prithvi.forward_features(x)

        # Last transformer layer
        tokens = features[-1]

        # Remove CLS token
        patch_tokens = tokens[:, 1:, :]

        B = patch_tokens.shape[0]

        # 3 temporal frames × 14 × 14 spatial patches
        patch_tokens = patch_tokens.reshape(
            B,
            3,
            14,
            14,
            768,
        )

        # Average the repeated temporal frames
        patch_tokens = patch_tokens.mean(dim=1)

        # [B, 14, 14, 768]
        # ->
        # [B, 768, 14, 14]
        feature_map = patch_tokens.permute(0, 3, 1, 2)

        x = self.conv1(feature_map)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.conv2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.conv3(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.conv4(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.classifier(x)

        return x


# ============================================================
# 7. LOAD PRITHVI MODEL
# ============================================================

def _build_prithvi_model():

    print("Building Prithvi-100M architecture...")

    backbone = PrithviMAE(
        img_size=224,
        patch_size=16,
        in_chans=6,
        num_frames=3,
        tubelet_size=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    )

    model = PrithviSegmentation(
        backbone,
        num_classes=5,
    )

    checkpoint = torch.load(
        str(PRITHVI_PATH),
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 8. LOAD U-NET FALLBACK MODEL
# ============================================================

def _build_unet_model():

    import segmentation_models_pytorch as smp

    print("Building U-Net fallback model...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=5,
    )

    checkpoint = torch.load(
        str(UNET_PATH),
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 9. MODEL LOADER
# ============================================================

def _load_model():

    global _model
    global _model_name

    if _model is not None:
        return _model

    # --------------------------------------------------------
    # Prefer fine-tuned Prithvi
    # --------------------------------------------------------

    if PRITHVI_PATH.exists():

        try:

            print()
            print("Loading preferred model:")
            print("  Prithvi:", PRITHVI_PATH)

            _model = _build_prithvi_model()
            _model_name = "Prithvi-100M"

            print("Loaded model: Prithvi-100M")

            return _model

        except Exception as e:

            print()
            print("WARNING: Prithvi model could not be loaded.")
            print("Reason:", e)
            print("Trying U-Net fallback...")

    # --------------------------------------------------------
    # U-Net fallback
    # --------------------------------------------------------

    if UNET_PATH.exists():

        _model = _build_unet_model()
        _model_name = "U-Net ResNet34"

        print("Loaded model: U-Net ResNet34")

        return _model

    raise FileNotFoundError(
        "Neither prithvi_finetuned.pth nor unet_best.pth was found."
    )


# ============================================================
# 10. LOAD SATELLITE IMAGE
# ============================================================

def _load_image(year):

    global _image_2023
    global _image_2015

    if year == 2023:

        if _image_2023 is None:

            if not IMAGE_2023_PATH.exists():
                raise FileNotFoundError(
                    f"2023 imagery not found: {IMAGE_2023_PATH}"
                )

            print("Loading 2023 satellite image...")

            _image_2023 = np.load(
                str(IMAGE_2023_PATH)
            )

        return _image_2023

    if year == 2015:

        if _image_2015 is None:

            if not IMAGE_2015_PATH.exists():
                raise FileNotFoundError(
                    f"2015 imagery not found: {IMAGE_2015_PATH}"
                )

            print("Loading 2015 satellite image...")

            _image_2015 = np.load(
                str(IMAGE_2015_PATH)
            )

        return _image_2015

    raise ValueError("Year must be 2015 or 2023.")


# ============================================================
# 11. GET 224 × 224 IMAGE PATCH
# ============================================================

def _get_image_patch(lat, lon, radius_m):

    """
    Extract a 224 × 224 × 6 Sentinel-2 patch
    around a latitude/longitude location.

    Returns:
        patch : (6, 224, 224)
        r1    : starting row
        c1    : starting column
    """

    if not (
        BBOX_MIN_LAT <= lat <= BBOX_MAX_LAT
        and
        BBOX_MIN_LON <= lon <= BBOX_MAX_LON
    ):

        raise ValueError(
            f"Location ({lat}, {lon}) is outside the "
            f"{STUDY_AREA} study-area BBOX."
        )

    image = _load_image(2023)

    bands, rows, cols = image.shape

    if bands != 6:
        raise ValueError(
            f"Expected 6 bands, but found {bands}."
        )

    # --------------------------------------------------------
    # Convert longitude → column
    # --------------------------------------------------------

    col_frac = (
        lon - BBOX_MIN_LON
    ) / (
        BBOX_MAX_LON - BBOX_MIN_LON
    )

    # --------------------------------------------------------
    # Convert latitude → row
    #
    # Raster row 0 corresponds approximately to the
    # northern/top part of the image.
    # Therefore latitude is inverted.
    # --------------------------------------------------------

    row_frac = (
        BBOX_MAX_LAT - lat
    ) / (
        BBOX_MAX_LAT - BBOX_MIN_LAT
    )

    c_centre = int(
        round(col_frac * (cols - 1))
    )

    r_centre = int(
        round(row_frac * (rows - 1))
    )

    # --------------------------------------------------------
    # Fixed model input size = 224 × 224
    # --------------------------------------------------------

    half = 112

    r1 = r_centre - half
    c1 = c_centre - half

    r1 = max(0, min(r1, rows - 224))
    c1 = max(0, min(c1, cols - 224))

    r2 = r1 + 224
    c2 = c1 + 224

    patch = image[
        :,
        r1:r2,
        c1:c2,
    ].astype(np.float32)

    if patch.shape != (6, 224, 224):

        raise ValueError(
            f"Unexpected patch shape: {patch.shape}"
        )

    # --------------------------------------------------------
    # Fill NaN values using band mean
    # --------------------------------------------------------

    for b in range(6):

        band = patch[b]

        if np.isnan(band).any():

            mean_value = np.nanmean(band)

            if not np.isfinite(mean_value):
                mean_value = 0.0

            patch[b] = np.where(
                np.isnan(band),
                mean_value,
                band,
            )

    return patch, r1, c1


# ============================================================
# 12. COMPUTE NDVI AND NDWI
# ============================================================

def _compute_ndvi_ndwi(patch):

    """
    Compute mean NDVI and NDWI from a
    (6, H, W) Sentinel-2 patch.

    Band order:
        0 = B2 Blue
        1 = B3 Green
        2 = B4 Red
        3 = B8 NIR
        4 = B11 SWIR1
        5 = B12 SWIR2
    """

    nir = patch[3].astype(np.float32)
    red = patch[2].astype(np.float32)
    green = patch[1].astype(np.float32)

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    ndvi_denominator = nir + red

    ndvi = np.where(
        ndvi_denominator != 0,
        (nir - red) / ndvi_denominator,
        0.0,
    )

    # --------------------------------------------------------
    # NDWI
    # --------------------------------------------------------

    ndwi_denominator = green + nir

    ndwi = np.where(
        ndwi_denominator != 0,
        (green - nir) / ndwi_denominator,
        0.0,
    )

    ndvi_mean = float(
        np.nanmean(ndvi)
    )

    ndwi_mean = float(
        np.nanmean(ndwi)
    )

    return ndvi_mean, ndwi_mean


# ============================================================
# 13. NDVI HUMAN-READABLE LABEL
# ============================================================

def _get_ndvi_label(ndvi):

    if ndvi > 0.5:

        return "Dense healthy forest/vegetation"

    elif ndvi > 0.3:

        return "Moderate healthy vegetation"

    elif ndvi > 0.1:

        return "Sparse or stressed vegetation"

    elif ndvi > 0.0:

        return "Very sparse vegetation / degraded land"

    else:

        return "Urban surface, water, or bare rock"


# ============================================================
# 14. CLASSIFY IMAGERY
# ============================================================

def classify_imagery(
    lat: float,
    lon: float,
    radius_m: int = 1000,
):

    """
    Classify satellite imagery around a location.

    Parameters
    ----------
    lat : float
        Latitude.

    lon : float
        Longitude.

    radius_m : int
        Requested area radius in metres.

    Returns
    -------
    dict
        Structured land-cover, spectral-index,
        confidence and change-detection result.
    """

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = _load_model()

    # --------------------------------------------------------
    # Get 2023 patch
    # --------------------------------------------------------

    patch, r1, c1 = _get_image_patch(
        lat,
        lon,
        radius_m,
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    if _model_name == "Prithvi-100M":

        # ----------------------------------------------------
        # Prithvi expects:
        # [B, 6, 3, 224, 224]
        # ----------------------------------------------------

        x = torch.tensor(
            patch,
            dtype=torch.float32,
        )

        # Clean data are approximately 0–1.
        # Prithvi normalization expects Sentinel-like values.
        x = x * 10000.0

        x = (
            x - PRITHVI_MEAN
        ) / PRITHVI_STD

        # Repeat one date three times
        x = x.unsqueeze(1).repeat(
            1,
            3,
            1,
            1,
        )

        # Add batch dimension
        x = x.unsqueeze(0)

    else:

        # ----------------------------------------------------
        # U-Net expects:
        # [B, 6, 224, 224]
        # ----------------------------------------------------

        x = torch.tensor(
            patch,
            dtype=torch.float32,
        ).unsqueeze(0)

    x = x.to(DEVICE)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(
            outputs,
            dim=1,
        )

        pred = outputs.argmax(
            dim=1
        )

    probs_np = probs.cpu().numpy()[0]
    pred_np = pred.cpu().numpy()[0]

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = {}

    for class_id, class_name in CLASSES.items():

        percentage = (
            np.mean(
                pred_np == class_id
            ) * 100.0
        )

        class_distribution[class_name] = round(
            float(percentage),
            1,
        )

    # --------------------------------------------------------
    # Dominant class
    # --------------------------------------------------------

    class_pixel_counts = np.bincount(
        pred_np.flatten(),
        minlength=5,
    )

    dominant_id = int(
        np.argmax(class_pixel_counts)
    )

    # --------------------------------------------------------
    # Confidence
    #
    # Mean probability of the dominant class
    # across the complete patch.
    # --------------------------------------------------------

    confidence = float(
        probs_np[dominant_id].mean()
        * 100.0
    )

    # --------------------------------------------------------
    # NDVI / NDWI
    # --------------------------------------------------------

    ndvi, ndwi = _compute_ndvi_ndwi(
        patch
    )

    ndvi_label = _get_ndvi_label(
        ndvi
    )

    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    change_flag = False
    ndvi_change = 0.0

    old_image = _load_image(2015)

    old_patch = old_image[
        :,
        r1:r1 + 224,
        c1:c1 + 224,
    ].astype(np.float32)

    # Fill NaN in old patch
    for b in range(6):

        band = old_patch[b]

        if np.isnan(band).any():

            mean_value = np.nanmean(band)

            if not np.isfinite(mean_value):
                mean_value = 0.0

            old_patch[b] = np.where(
                np.isnan(band),
                mean_value,
                band,
            )

    old_ndvi, _ = _compute_ndvi_ndwi(
        old_patch
    )

    ndvi_change = abs(
        ndvi - old_ndvi
    )

    # Manual threshold:
    # absolute NDVI difference > 0.15
    # = significant change
    change_flag = bool(
        ndvi_change > 0.15
    )

    if change_flag:

        change_description = (
            "Significant land change detected "
            "2015→2023"
        )

    else:

        change_description = (
            "No significant change detected"
        )

    # --------------------------------------------------------
    # Inference time
    # --------------------------------------------------------

    inference_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "latitude":
            round(float(lat), 6),

        "longitude":
            round(float(lon), 6),

        "radius_m":
            int(radius_m),

        "land_cover":
            CLASSES[dominant_id],

        "class_id":
            dominant_id,

        "confidence_pct":
            round(confidence, 1),

        "ndvi":
            round(ndvi, 3),

        "ndwi":
            round(ndwi, 3),

        "ndvi_label":
            ndvi_label,

        "class_distribution":
            class_distribution,

        "change_flag":
            change_flag,

        "change_description":
            change_description,

        "ndvi_2015":
            round(old_ndvi, 3),

        "ndvi_change":
            round(ndvi_change, 3),

        "model":
            _model_name,

        "inference_time_sec":
            round(inference_time, 3),
    }

    return result


# ============================================================
# 15. MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("GeoSense Agent 2.0 - Exercise 5")
    print("image_classifier.py")
    print("=" * 65)

    print()
    print("Study area :", STUDY_AREA)

    print(
        "BBOX       :",
        BBOX_MIN_LON,
        BBOX_MIN_LAT,
        "to",
        BBOX_MAX_LON,
        BBOX_MAX_LAT,
    )

    print(
        "Device     :",
        DEVICE,
    )

    print()
    print("Loading and testing classifier...")
    print()

    # Mumbai BBOX centre
    test_lat = (
        BBOX_MIN_LAT + BBOX_MAX_LAT
    ) / 2.0

    test_lon = (
        BBOX_MIN_LON + BBOX_MAX_LON
    ) / 2.0

    print(
        f"Test location: "
        f"{test_lat:.6f}, "
        f"{test_lon:.6f}"
    )

    try:

        result = classify_imagery(
            test_lat,
            test_lon,
            radius_m=500,
        )

        print()
        print("-" * 65)
        print("CLASSIFICATION RESULT")
        print("-" * 65)

        for key, value in result.items():

            print(
                f"{key:25s}: {value}"
            )

        print("-" * 65)

        if result["inference_time_sec"] < 3:

            print(
                "Inference time check : PASS (< 3 sec)"
            )

        else:

            print(
                "Inference time check : "
                "ABOVE 3 sec"
            )

    except Exception as e:

        print()
        print("ERROR:")
        print(type(e).__name__, e)

        raise