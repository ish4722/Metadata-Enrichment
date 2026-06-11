from huggingface_hub import snapshot_download
snapshot_download("jinaai/jina-embeddings-v3", local_dir="./jina-v3")