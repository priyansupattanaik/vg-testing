import os


def _setup_hf_token():
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        pass


def setup_cache():
    base = "/content/drive/MyDrive/visionguard_cache"
    _setup_hf_token()
    if not os.path.exists("/content/drive/MyDrive"):
        return
    paths = {
        "HF_HOME": os.path.join(base, "hf"),
        "TRANSFORMERS_CACHE": os.path.join(base, "hf", "transformers"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(base, "hf", "hub"),
        "TORCH_HOME": os.path.join(base, "torch"),
        "YOLO_CONFIG_DIR": os.path.join(base, "ultralytics"),
        "ULTRALYTICS_SETTINGS": os.path.join(base, "ultralytics", "settings.json"),
    }
    for key, path in paths.items():
        os.environ.setdefault(key, path)
    for k in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "YOLO_CONFIG_DIR"]:
        os.makedirs(os.environ[k], exist_ok=True)
