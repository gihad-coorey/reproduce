import os
from huggingface_hub import snapshot_download

TARGET = "data/libero"

snapshot_download(
	repo_id="lerobot/libero",
	local_dir=TARGET,
	repo_type="dataset"
)

print("LIBERO dataset downloaded to", TARGET)
