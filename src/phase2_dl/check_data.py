import numpy as np


for year in ["2015", "2023"]:

    print(f"\n========== {year} ==========")

    # Load CLEAN image chips
    images = np.load(
        f"data/satellite/chips/chips_{year}_clean.npy"
    )

    # Load ORIGINAL LABEL chips
    labels = np.load(
        f"data/satellite/chips/label_chips_{year}.npy"
    )

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    print("NaN values:", np.isnan(images).sum())
    print("Infinite values:", np.isinf(images).sum())

    print("Minimum image value:", np.min(images))
    print("Maximum image value:", np.max(images))
    print("Mean image value:", np.mean(images))

    print("Unique label classes:", np.unique(labels))


print("\nData check completed!")