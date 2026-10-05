import torch
import numpy as np
import os
import sys

# Import the architecture and helpers from your training script
# Ensure 'GNNAE.py' is in the same directory
try:
    from .GNNAE import Graph2Seq, collate_fn_gnn, parse_sexpr, device
except ImportError:
    print("Error: Could not import from 'GNNAE.py'. Make sure the file exists.")
    sys.exit(1)

class FHEFeatureExtractor:
    def __init__(self, model_path, device=device):
        self.device = device
        print(f"[FHE Embedding] Loading model from {model_path}...")

        # 1. Initialize the Model Architecture
        self.model = Graph2Seq()

        # 2. Check path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # 3. Load Weights
        # Map to the specified device (cpu or cuda)
        state_dict = torch.load(model_path, map_location=self.device)

        # Fix DDP 'module.' prefix if the model was trained on multi-GPU
        new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(new_sd)
        self.model.to(self.device)
        self.model.eval()

        # 4. Freeze Weights (Critical for RL stability)
        for param in self.model.parameters():
            param.requires_grad = False

        print("[FHE Embedding] Model loaded and frozen successfully.")

    def get_embedding(self, expr_str):
        """
        Input: S-expression string e.g. "(VecAdd (Vec a b) (Vec 1 1))"
        Output: Numpy array shape (256,)
        """
        try:
            # A. Parse the string into an Expr tree
            expr_tree = parse_sexpr(expr_str)

            # B. Convert to Graph Batch (Batch size 1)
            # collate_fn_gnn returns (batch_data, targets), we only need batch_data
            batch_data, _ = collate_fn_gnn([expr_tree])
            batch_data = batch_data.to(self.device)

            # C. Run Inference
            with torch.no_grad():
                # Extract the 256-dim vector from the GNN Encoder
                embedding = self.model.get_embedding(batch_data)

            # D. Convert to Numpy for Gym
            return embedding.cpu().numpy()[0].astype(np.float32)

        except Exception as e:
            # Fallback if parsing fails (prevents RL crash)
            print(f"[Embedding Error] Failed on: {expr_str[:50]}... | Error: {e}")
            return np.zeros(256, dtype=np.float32)
