import os
from pathlib import Path
import torch
import enum

class ConstraintMethod(enum.Enum):
    NONE = "none"
    LAGRANGIAN_OD_OV = "lagrangian_od_ov"
    LAGRANGIAN_PERSTEP = "lagrangian_perstep"
    LAGRANGIAN_ALWAYS_DONE = "lagrangian_always_done"
    MARGIN_BARRIER = "margin_barrier"
    NOISE_MASKING = "noise_masking"
    NATO_SC = "nato_sc"
    LAGRANGIAN_PID = "lagrangian_pid"

PROJECT_ROOT = Path(__file__).parent.parent
FHE_RL_DIR = Path(__file__).parent
# A l'intérieur de config.py

MODEL_PATHS = {
   #"agent_model": PROJECT_ROOT / "checkpoints" / "model_L1_none_film_a" / "rl_model_100000_steps.zip",
     "agent_model": PROJECT_ROOT / "checkpoints" / "model_jobid_lagrangian_pid_film_a" / "rl_model_600000_steps.zip",
    "gnn_embeddings_model": PROJECT_ROOT / "trained_models" / "embeddings_gnn_model_epoch_100.pth",
}

AGENT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def get_model_path(model_key):
    primary_path = MODEL_PATHS.get(model_key)
    if primary_path and primary_path.exists(): return str(primary_path)
    elif primary_path: return str(primary_path)
    else: raise KeyError(f"Unknown model key: {model_key}")

def get_device(): return AGENT_CONFIG["device"]

def print_config():
    print("=== FHE RL Agent Configuration ===")
    for key, path in MODEL_PATHS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {key}: {status} {path}")
