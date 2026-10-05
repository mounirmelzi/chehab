import os
import builtins
import sys 
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# --- PyTorch Geometric ---
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
except ImportError:
    print("Error: PyTorch Geometric not installed. pip install torch_geometric torch_scatter torch_sparse")
    sys.exit(1)

# --- Custom Imports ---
from .utils  import load_expressions
from pytrs import (
    Op, Const, Var, VARIABLE_RANGE, CONST_OFFSET,
    PAREN_CLOSE, PAREN_OPEN, node_to_id, parse_sexpr,
    tokenize, MAX_INT_TOKENS
)

# ------------------------------------------------------------------
# 1. Setup & DDP
# ------------------------------------------------------------------
torch.set_float32_matmul_precision("high")
# Disable expandable_segments to fix the warning/potential OOM trigger
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

def print(*args, **kwargs):  
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    if not torch.cuda.is_available(): sys.exit(1)
    if not dist.is_initialized(): dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    
    num_gpus = torch.cuda.device_count()
    if ddp_local_rank >= num_gpus: sys.exit(1)
    
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    if master_process:
        print(f"DDP Init: {ddp_world_size} GPUs")
else:
    master_process = True
    ddp_rank = 0
    ddp_world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if master_process:
    print(f"Device: {device}")

# ------------------------------------------------------------------
# 2. Config & Constants
# ------------------------------------------------------------------
TYPE_PLAINTEXT = 1   
TYPE_CIPHERTEXT = 2  

MAX_FHE_DEPTH = 64    
MAX_TOPO_DEPTH = 200  
MAX_ARG_INDEX = 128   

class Config:
    # 1. Calculate Base Vocab Size from pytrs
    base_vocab_size = CONST_OFFSET + MAX_INT_TOKENS + 5 
    
    # 2. Define Special Tokens relative to base
    start_token = base_vocab_size
    end_token = base_vocab_size + 1
    pad_token = base_vocab_size + 2
    
    # 3. Define Generic IDs (THE FIX: Make them part of the vocab)
    generic_const_id = base_vocab_size + 3
    generic_var_id = base_vocab_size + 4
    
    # 4. Final Vocab Size
    vocab_size = base_vocab_size + 5
    
    # --- MODEL SPECS ---
    d_model = 256          
    num_heads = 8          
    gnn_layers = 8         
    
    # Decoder
    num_decoder_layers = 4
    dim_feedforward = 1024 
    dropout = 0.1
    
    max_gen_length = 512
    batch_size = 128
    learning_rate = 3e-4   
    epochs = 100

config = Config()

# ------------------------------------------------------------------
# 3. Position-Aware Graph Builder
# ------------------------------------------------------------------
class GraphBuilder:
    def _analyze_node(self, node):
        if isinstance(node, Const) or isinstance(node, (int, float)):
            return TYPE_PLAINTEXT, 0, 0
        if isinstance(node, Var) or isinstance(node, str):
            return TYPE_CIPHERTEXT, 0, 0

        if isinstance(node, Op):
            child_results = [self._analyze_node(arg) for arg in node.args]
            types = [r[0] for r in child_results]
            fhe_depths = [r[1] for r in child_results]
            topo_depths = [r[2] for r in child_results]

            current_type = TYPE_CIPHERTEXT if (TYPE_CIPHERTEXT in types) else TYPE_PLAINTEXT
            current_topo = (max(topo_depths) + 1) if topo_depths else 0
            max_fhe = max(fhe_depths) if fhe_depths else 0
            
            is_mul = (node.op in ["VecMul", "*", "Mul"])
            if is_mul and current_type == TYPE_CIPHERTEXT:
                ct_children = sum(1 for t in types if t == TYPE_CIPHERTEXT)
                current_fhe = max_fhe + 1 if ct_children >= 2 else max_fhe
            else:
                current_fhe = max_fhe

            return (current_type, min(current_fhe, MAX_FHE_DEPTH), min(current_topo, MAX_TOPO_DEPTH))
        return 0, 0, 0

    def expr_to_graph(self, expr):
        node_ids, node_types, node_fhe, node_topo, node_args = [], [], [], [], []
        edge_src, edge_dst = [], []
        
        def get_token_id(node):
            # 1. Operators: Keep Identity
            if isinstance(node, Op):
                nid, _, _, _ = node_to_id(node, {}, {}, 0, 0)
                # Safety check for bounds
                return min(nid, config.vocab_size - 1)
            
            # 2. Constants: Normalize to Generic (FIXED ID)
            if isinstance(node, Const) or isinstance(node, (int, float)):
                return config.generic_const_id
                
            # 3. Variables: Normalize to Generic (FIXED ID)
            if isinstance(node, Var) or isinstance(node, str):
                return config.generic_var_id
            return 0

        def traverse(node, arg_idx=0):
            d_type, d_fhe, d_topo = self._analyze_node(node)
            token_id = get_token_id(node)
            
            curr_idx = len(node_ids)
            node_ids.append(token_id)
            node_types.append(d_type)
            node_fhe.append(d_fhe)
            node_topo.append(d_topo)
            node_args.append(min(arg_idx, MAX_ARG_INDEX))
            
            if isinstance(node, Op):
                for i, child in enumerate(node.args):
                    c_idx = traverse(child, arg_idx=i)
                    edge_src.append(c_idx)
                    edge_dst.append(curr_idx)
            return curr_idx

        root_idx = traverse(expr, 0)
        
        x = torch.stack([
            torch.tensor(node_ids), 
            torch.tensor(node_types),
            torch.tensor(node_fhe), 
            torch.tensor(node_topo),
            torch.tensor(node_args)
        ], dim=1).long()
        
        edge_index = torch.tensor([edge_src, edge_dst]).long()
        return Data(x=x, edge_index=edge_index, root_index=root_idx)

    def flatten_for_tgt(self, expr):
        l = []
        def dfs(e):
            if isinstance(e, Op):
                l.append(PAREN_OPEN)
                nid, _, _, _ = node_to_id(e, {}, {}, 0, 0)
                l.append(nid)
                for c in e.args: dfs(c)
                l.append(PAREN_CLOSE)
            else:
                nid, _, _, _ = node_to_id(e, {}, {}, 0, 0)
                l.append(nid)
        dfs(expr)
        return l

graph_builder = GraphBuilder()

# ------------------------------------------------------------------
# 4. GNN Encoder (With Positional Embedding)
# ------------------------------------------------------------------
class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.type_emb = nn.Embedding(4, config.d_model)
        self.fhe_depth_emb = nn.Embedding(MAX_FHE_DEPTH + 1, config.d_model)
        self.topo_depth_emb = nn.Embedding(MAX_TOPO_DEPTH + 1, config.d_model)
        self.arg_pos_emb = nn.Embedding(MAX_ARG_INDEX + 1, config.d_model)
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(config.gnn_layers):
            self.layers.append(GATv2Conv(config.d_model, config.d_model, heads=config.num_heads, concat=False, dropout=0.1))
            self.norms.append(nn.LayerNorm(config.d_model))
            
        self.final_norm = nn.LayerNorm(config.d_model)

        self.compressor = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Tanh()
        )

    def forward(self, x, edge_index, batch_vec, root_indices):
        h = (self.token_emb(x[:, 0]) + 
             self.type_emb(x[:, 1]) + 
             self.fhe_depth_emb(x[:, 2]) + 
             self.topo_depth_emb(x[:, 3]) +
             self.arg_pos_emb(x[:, 4])) 
        
        for conv, norm in zip(self.layers, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = norm(h)
            h = h + h_in
        h = self.final_norm(h)

        h_root = h[root_indices]
        h_mean = global_mean_pool(h, batch_vec)
        h_max = global_max_pool(h, batch_vec)
        
        h_cat = torch.cat([h_root, h_mean, h_max], dim=1)
        return self.compressor(h_cat)

# ------------------------------------------------------------------
# 5. Decoder
# ------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_gen_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.output_fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, memory, tgt_seq):
        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        tgt_padding_mask = (tgt_seq == config.pad_token)

        memory = memory.unsqueeze(1) 

        dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.output_fc(dec_out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# ------------------------------------------------------------------
# 6. Wrapper
# ------------------------------------------------------------------
class Graph2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GNNEncoder()
        self.decoder = TransformerDecoder()

    def forward(self, batch_data, tgt_seq):
        root_indices = batch_data.ptr[1:] - 1
        graph_emb = self.encoder(batch_data.x, batch_data.edge_index, batch_data.batch, root_indices)
        logits = self.decoder(graph_emb, tgt_seq)
        return logits
    
    def get_embedding(self, batch_data):
        root_indices = batch_data.ptr[1:] - 1
        return self.encoder(batch_data.x, batch_data.edge_index, batch_data.batch, root_indices)

# ------------------------------------------------------------------
# 7. Training
# ------------------------------------------------------------------
def collate_fn_gnn(batch_exprs):
    data_list, tgt_seqs = [], []
    for expr in batch_exprs:
        data_list.append(graph_builder.expr_to_graph(expr))
        flat_ids = graph_builder.flatten_for_tgt(expr)
        flat_ids = flat_ids[: config.max_gen_length - 2]
        tgt_seqs.append([config.start_token] + flat_ids + [config.end_token])
        
    batch_data = Batch.from_data_list(data_list)
    padded_tgt = [t + [config.pad_token] * (config.max_gen_length - len(t)) for t in tgt_seqs]
    return batch_data, torch.tensor(padded_tgt, dtype=torch.long)

def train(model, dataset):
    train_sampler = DistributedSampler(dataset, drop_last=True) if ddp else None
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn_gnn,
        num_workers=4,
        pin_memory=True
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)
    opt = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)
    
    if master_process:
        print(f"Position-Aware GNN Training (Root+Mean+Max -> {config.d_model})")

    for epoch in range(config.epochs):
        model.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        total_loss = 0
        batch_count = 0
        
        for i, (batch_data, tgt_seq) in enumerate(train_loader):
            batch_data, tgt_seq = batch_data.to(device), tgt_seq.to(device)
            opt.zero_grad()
            logits = model(batch_data, tgt_seq[:, :-1])
            loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_seq[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            batch_count += 1
            if master_process: print(f"[E{epoch} B{i}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        scheduler.step(avg_loss)
        
        if master_process:
            print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} | LR: {opt.param_groups[0]['lr']}")
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/gnn_pos_latest_n.pth")
            torch.save(model.state_dict(), f"saved_models/gnn_pos_epoch_{epoch}_n.pth")

# ------------------------------------------------------------------
# 8. Interface
# ------------------------------------------------------------------
def get_vector_for_rl(model, expr_str):
    model.eval()
    expr = parse_sexpr(expr_str)
    batch_data, _ = collate_fn_gnn([expr])
    batch_data = batch_data.to(device)
    with torch.no_grad():
        embedding = model.get_embedding(batch_data)
    return embedding.cpu().numpy()[0]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    model = Graph2Seq().to(device)
    if ddp: model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    if mode == "train":
        all_expressions = load_expressions("./pretraining/dataset_balanced_ROT_32_15_5000000.txt")
        all_expressions = sorted(all_expressions, key=len, reverse=True)
        all_expressions = list(map(parse_sexpr, all_expressions))
        train(model, all_expressions)
        if ddp:
            dist.barrier()
            dist.destroy_process_group()

    elif mode == "test_rl":
        model_path = "saved_models/gnn_pos_latest_n.pth"
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
            print("Model loaded.")
        
        print("--- Testing Positional Awareness ---")
        expr1 = "(VecAdd (Vec a b) (Vec c d))"
        expr2 = "(VecMinus (Vec a b) (Vec c d))"
        v1 = get_vector_for_rl(model, expr1)
        v2 = get_vector_for_rl(model, expr2)
        dist = np.linalg.norm(v1 - v2)
        print(f"Swap Distance (should be > 0.0): {dist:.5f}")
        
        print("\n--- Testing Variable Invariance ---")
        expr3 = "(Vec (* a b))"
        expr4 = "(Vec (* x y))"
        v3 = get_vector_for_rl(model, expr3)
        v4 = get_vector_for_rl(model, expr4)
        dist_var = np.linalg.norm(v3 - v4)
        print(f"Var Rename Dist (should be 0.0): {dist_var:.5f}")
