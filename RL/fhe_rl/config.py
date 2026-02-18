"""
Configuration file for FHE RL Agent
Contains model paths, tokenizer settings, and other configuration parameters
"""

import os
from pathlib import Path
import torch
import enum


class RLAlgorithm(enum.Enum):
    PPO = "PPO"
    LAGRANGIAN_PPO = "LAGRANGIAN_PPO"


class ConstraintMethod(enum.Enum):
    """Constraint enforcement method for multi-budget training."""
    NONE = "none"                                # Pure PPO, no constraint
    LAGRANGIAN_OD_OV = "lagrangian_od_ov"        # ON_DONE + ON_VIOLATION (current stable)
    LAGRANGIAN_PERSTEP = "lagrangian_perstep"    # Per-step violation penalty
    LAGRANGIAN_ALWAYS_DONE = "lagrangian_always_done"  # Always penalize at terminal
    MARGIN_BARRIER = "margin_barrier"            # Margin obs + hard terminal penalty (NEW)


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up to RL/ directory
FHE_RL_DIR = Path(__file__).parent

# Model paths configuration
MODEL_PATHS = {
    "agent_model": FHE_RL_DIR / "trained_models" / "agent_dynamic_llm_data.zip",
    "dynamic_embeddings_model": FHE_RL_DIR / "trained_models" / "embeddings_ROT_15_32_5m_10742576.pth",
    "bpe_embeddings_model": FHE_RL_DIR / "trained_models" / "model_Transformer_BPE_ddp_jobid_epoch_5000000.pth",
    "bpe_tokenizer": FHE_RL_DIR / "trained_models" / "bpe_tokenizer.pkl",
}

# Tokenizer configuration
TOKENIZER_CONFIG = {
    "use_bpe": False,  # Set to True to use BPE tokenizer, False for dynamic tokenizer
    "default_vocab_size": 1000,  # Default vocab size (not used but just for safety)"
    "auto_detect_vocab": True,   # Automatically detect vocab size from model
}

# Agent configuration
AGENT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "algorithm": RLAlgorithm.LAGRANGIAN_PPO,
}

def get_model_path(model_key):
    """
    Get the path for a specific model
    """
    primary_path = MODEL_PATHS.get(model_key)

    if primary_path and primary_path.exists():
        return str(primary_path)
    else:
        raise FileNotFoundError(f"Model file not found for {model_key}. Checked path: {primary_path}")

def get_tokenizer_type():
    """
    Get the configured tokenizer type
    """
    return "bpe" if TOKENIZER_CONFIG["use_bpe"] else "dynamic"

def set_tokenizer_type(tokenizer_type):
    """
    Set the tokenizer type to use
    """
    if tokenizer_type not in ["dynamic", "bpe"]:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}. Must be 'dynamic' or 'bpe'")
    
    TOKENIZER_CONFIG["use_bpe"] = (tokenizer_type == "bpe")

def get_vocab_size():
    """
    Get the configured vocab size
    """
    return TOKENIZER_CONFIG["default_vocab_size"]

def get_device():
    """
    Get the configured device
    """
    return AGENT_CONFIG["device"]

def get_rl_algorithm() -> RLAlgorithm:
    """
    Get the configured RL algorithm
    """
    return AGENT_CONFIG["algorithm"]

def print_config():
    """
    Print the current configuration
    """
    print("=== FHE RL Agent Configuration ===")
    print(f"Tokenizer type: {get_tokenizer_type()}")
    print(f"Default vocab size: {get_vocab_size()}")
    print(f"Device: {get_device()}")
    print(f"RL Algorithm: {get_rl_algorithm().value}")
    print("\nModel paths:")
    for key, path in MODEL_PATHS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {key}: {status} {path}")

