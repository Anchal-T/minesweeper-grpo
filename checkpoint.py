"""Small resumable LoRA checkpoints, optionally mirrored to the Hub."""
import os
import random
import time

import torch


def restore_from_hub(path, repo_id):
    if os.path.exists(os.path.join(path, "training.pt")) or not repo_id:
        return
    from huggingface_hub import snapshot_download
    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=path,
                      token=os.environ.get("HF_TOKEN"))


def save_checkpoint(path, model, optimizer, scaler, step, repo_id=None):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, os.path.join(path, "training.pt"))
    if repo_id and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=path,
                          commit_message=f"training checkpoint step {step}")


def load_training_state(path, optimizer, scaler, device):
    state = torch.load(os.path.join(path, "training.pt"),
                       map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def time_budget_expired(started, minutes):
    return minutes is not None and time.monotonic() - started >= minutes * 60
