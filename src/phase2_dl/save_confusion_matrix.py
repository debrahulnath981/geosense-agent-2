import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# GeoSense Phase 2
# Save Prithvi Confusion Matrix
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Confusion matrix from Prithvi validation evaluation
cm = np.array([
    [0,      818,     93,  23981,   6557],
    [0, 1683667,    343,  21111, 316366],
    [0,     281, 1471442, 29680,    574],
    [0,   34315,  52585, 500914, 291259],
    [0,  222493,   1308, 158772, 752977]
])

classes = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
csv_file = OUTPUT_DIR / "prithvi_confusion_matrix.csv"

df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

df.to_csv(csv_file)

print("Saved CSV:")
print(csv_file)

# ------------------------------------------------------------
# Save PNG
# ------------------------------------------------------------
png_file = OUTPUT_DIR / "prithvi_confusion_matrix.png"

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(cm)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Prithvi-100M Confusion Matrix")

# Add numbers
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax, label="Pixel Count")

plt.tight_layout()
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved PNG:")
print(png_file)

# ------------------------------------------------------------
# Basic verification
# ------------------------------------------------------------
print("\nConfusion Matrix:")
print(cm)

print("\nTotal validation pixels:", cm.sum())

print("\nExpected total: 5,569,536")

if cm.sum() == 5569536:
    print("TOTAL CHECK: PASS")
else:
    print("TOTAL CHECK: CHECK REQUIRED")