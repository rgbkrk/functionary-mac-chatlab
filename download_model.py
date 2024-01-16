#from transformers import AutoModel
#model = AutoModel.from_pretrained("meetkai/functionary-7b-v2-GGUF")

from huggingface_hub import hf_hub_download

# Model repository on the Hugging Face model hub
model_repo = "meetkai/functionary-7b-v2-GGUF"

# File to download
file_name = "functionary-7b-v2.f16.gguf"

# Download the file
local_file_path = hf_hub_download(repo_id=model_repo, filename=file_name)

print(f"Downloaded file to {local_file_path}")