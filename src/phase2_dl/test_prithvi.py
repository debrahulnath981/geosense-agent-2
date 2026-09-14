from huggingface_hub import hf_hub_download

MODEL_ID = "ibm-nasa-geospatial/Prithvi-100M"

print("Testing Prithvi-100M download...")
print("Model:", MODEL_ID)

try:
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="Prithvi_100M.pt"
    )

    print("\nPrithvi download successful!")
    print("Model path:", model_path)

except Exception as e:

    print("\nPrithvi download failed.")
    print("Error:", e)