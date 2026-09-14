import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "ibm-nasa-geospatial/Prithvi-100M"

print("Inspecting Prithvi-100M checkpoint...\n")

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="Prithvi_100M.pt"
)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded.")
print("Number of parameters:", len(checkpoint))

print("\nFirst 30 checkpoint keys:")
for key in list(checkpoint.keys())[:30]:
    print(key)

print("\nEncoder-related keys:")
encoder_keys = [
    key for key in checkpoint.keys()
    if key.startswith("encoder.")
]

print("Number of encoder keys:", len(encoder_keys))

for key in encoder_keys[:10]:
    print(key)

print("\nSample parameter shapes:")

for key in [
    "encoder.cls_token",
    "encoder.pos_embed",
    "encoder.patch_embed.proj.weight",
    "encoder.patch_embed.proj.bias"
]:

    if key in checkpoint:
        print(key, "->", tuple(checkpoint[key].shape))

print("\nPrithvi checkpoint inspection completed successfully!")