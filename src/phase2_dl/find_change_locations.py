import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# GeoSense Phase 2
# Find Changed and Unchanged Locations from NDVI
# ============================================================

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

NDVI_2015 = PROJECT_ROOT / "data" / "satellite" / "clean" / "ndvi_2015.npy"
NDVI_2023 = PROJECT_ROOT / "data" / "satellite" / "clean" / "ndvi_2023.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "change_detection_candidates.csv"

# -----------------------------
# Mumbai BBOX
# -----------------------------
MIN_LON = 72.776333
MIN_LAT = 18.8939564
MAX_LON = 72.9806446
MAX_LAT = 19.2701767

# Change threshold from Exercise 5.2
CHANGE_THRESHOLD = 0.15

# Chip size = 224 x 224
PATCH_SIZE = 224
HALF = PATCH_SIZE // 2

# -----------------------------
# Load NDVI
# -----------------------------
print("Loading NDVI data...")

ndvi15 = np.load(NDVI_2015)
ndvi23 = np.load(NDVI_2023)

print("2015 shape:", ndvi15.shape)
print("2023 shape:", ndvi23.shape)

# -----------------------------
# Check shape
# -----------------------------
if ndvi15.shape != ndvi23.shape:
    raise ValueError("2015 and 2023 NDVI shapes are different!")

H, W = ndvi15.shape

print("Raster size:", H, "rows x", W, "columns")

# -----------------------------
# Find candidates
# -----------------------------
changed = None
unchanged = None

print("\nSearching for locations...")

# Avoid raster edges
for row in range(HALF, H - HALF, PATCH_SIZE):
    for col in range(HALF, W - HALF, PATCH_SIZE):

        r1 = row - HALF
        r2 = row + HALF

        c1 = col - HALF
        c2 = col + HALF

        patch15 = ndvi15[r1:r2, c1:c2]
        patch23 = ndvi23[r1:r2, c1:c2]

        mean15 = float(np.nanmean(patch15))
        mean23 = float(np.nanmean(patch23))

        difference = abs(mean23 - mean15)

        # Convert pixel position to geographic coordinates
        lat = MAX_LAT - (row / (H - 1)) * (MAX_LAT - MIN_LAT)
        lon = MIN_LON + (col / (W - 1)) * (MAX_LON - MIN_LON)

        result = {
            "row": row,
            "column": col,
            "latitude": lat,
            "longitude": lon,
            "ndvi_2015": mean15,
            "ndvi_2023": mean23,
            "ndvi_change": difference,
        }

        # Changed location
        if changed is None and difference > CHANGE_THRESHOLD:
            changed = result

        # Unchanged location
        if unchanged is None and difference <= CHANGE_THRESHOLD:
            unchanged = result

        # Stop once both are found
        if changed is not None and unchanged is not None:
            break

    if changed is not None and unchanged is not None:
        break

# -----------------------------
# Print results
# -----------------------------
print("\n========================================")
print("CHANGE DETECTION CANDIDATES")
print("========================================")

if changed is not None:
    print("\nCHANGED LOCATION")
    print("-------------------------")
    for key, value in changed.items():
        print(f"{key}: {value}")
else:
    print("\nNo changed location found.")

if unchanged is not None:
    print("\nUNCHANGED LOCATION")
    print("-------------------------")
    for key, value in unchanged.items():
        print(f"{key}: {value}")
else:
    print("\nNo unchanged location found.")

# -----------------------------
# Save CSV
# -----------------------------
rows = []

if changed is not None:
    changed["status"] = "Changed"
    rows.append(changed)

if unchanged is not None:
    unchanged["status"] = "Unchanged"
    rows.append(unchanged)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)