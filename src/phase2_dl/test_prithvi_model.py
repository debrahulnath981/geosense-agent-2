import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "ibm-nasa-geospatial/Prithvi-100M"

print("Testing Prithvi-100M model loading...")

try:
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="Prithvi_100M.pt"
    )

    print("Model file found:")
    print(model_path)

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    print("\nPrithvi checkpoint loaded successfully!")

    print("Checkpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):
        print("\nCheckpoint keys:")
        print(list(checkpoint.keys())[:20])

except Exception as e:

    print("\nPrithvi model loading failed.")
    print("Error:", e)