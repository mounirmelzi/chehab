
import sys
import os
import argparse
from .run import run_agent
from .train import train_agent
from .test import test_agent, test_agent_v2
from .utils import load_embeddings
from .TRAE_bpe import BPETokenizer  # Import for pickle compatibility
from .config import (
    get_model_path, get_tokenizer_type, 
    print_config
)


def parse_arguments(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FHE RL Agent")
    
    # Add tokenizer type argument
    parser.add_argument(
        '--tokenizer_type', 
        choices=['dynamic', 'bpe'], 
        default=get_tokenizer_type(),
        help='Tokenizer type to use (default: from config)'
    )
    
    # Add config flag
    parser.add_argument(
        '--show_config', 
        action='store_true',
        help='Show current configuration and exit'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument(
        '--budgets',
        type=str,
        default=None,
        help='Comma-separated list of noise budgets (e.g., "300,400,500,1000,9000000")'
    )
    train_parser.add_argument(
        '--timesteps',
        type=int,
        default=2_000_000,
        help='Total training timesteps (default: 2000000)'
    )
    train_parser.add_argument(
        '--num_envs',
        type=int,
        default=8,
        help='Number of parallel environments (default: 8)'
    )
    train_parser.add_argument(
        '--denom_factor',
        type=int,
        default=4,
        help='Lagrangian update frequency denominator (default: 4, lower=more frequent updates)'
    )
    train_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_od_ov',
        choices=['none', 'lagrangian_od_ov', 'lagrangian_perstep', 'lagrangian_always_done', 'margin_barrier', 'noise_masking', 'nato_sc'],
        help='Constraint enforcement method (default: lagrangian_od_ov)'
    )
    train_parser.add_argument(
        '--algo',
        type=str,
        default='ppo',
        choices=['ppo', 'focops', 'lagrangian_pid'],
        help='RL algorithm (default: ppo)'
    )
    train_parser.add_argument(
        '--budget_encoding',
        type=str,
        default='raw',
        choices=['raw', 'embed', 'film'],
        help='Budget encoding in policy (raw=one-hot, embed=learned embedding, film=FiLM conditioning)'
    )
    train_parser.add_argument(
        '--ent_coef',
        type=float,
        default=0.01,
        help='Entropy coefficient for exploration (default: 0.01)'
    )
    train_parser.add_argument(
        '--curriculum',
        action='store_true',
        default=False,
        help='Enable curriculum budget scheduling (tight budgets first, then widen)'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the agent')
    test_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model .zip file (e.g., model_14007517_lagrangian_od_ov.zip)'
    )
    test_parser.add_argument(
        '--budgets',
        type=str,
        default=None,
        help='Comma-separated budgets to TEST on (e.g., "240,300,1000000")'
    )
    test_parser.add_argument(
        '--train_budgets',
        type=str,
        default=None,
        help='Comma-separated budgets the model was TRAINED on (for correct obs space). Defaults to --budgets.'
    )
    test_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_od_ov',
        choices=['none', 'lagrangian_od_ov', 'lagrangian_perstep', 'lagrangian_always_done', 'margin_barrier', 'noise_masking', 'nato_sc'],
        help='Constraint method the model was trained with (default: lagrangian_od_ov)'
    )
    test_parser.add_argument(
        '--test_mode',
        type=str,
        default='v1',
        choices=['v1', 'v2'],
        help='v1=existing test, v2=trajectory checkpointing + safety rollback'
    )
    test_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output Excel file path (default: auto-generated from model name)'
    )
    test_parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        help='Path to benchmark expressions file (default: ./fhe_rl/datasets/benchmarks.txt)'
    )
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the agent')
    run_parser.add_argument('input_expr_file', help='Input expression file')
    run_parser.add_argument('output_vector_file', help='Output vector file')
    
    return parser.parse_args(args)


def usage() -> None:
    print(
        "Usage:\n"
        "  python -m fhe_rl train [--tokenizer_type {dynamic,bpe}]\n"
        "  python -m fhe_rl test  [--tokenizer_type {dynamic,bpe}]\n"
        "  python -m fhe_rl run   [--tokenizer_type {dynamic,bpe}] "
        "<input_expr_file> <output_vector_file>\n"
        "  python -m fhe_rl --show_config  # Show current configuration\n"
        "\n"
        "Options:\n"
        "  --tokenizer_type {dynamic,bpe}  Choose tokenizer type (overrides config)\n"
        "  --show_config                   Show current configuration\n"
        "\n"
        "All model paths are loaded from config.py."
    )
    sys.exit(1)


def load_embeddings_from_config(tokenizer_type=None):
    """Load embeddings using the configuration system"""
    try:
        # Determine the correct embeddings model based on tokenizer type
        if tokenizer_type == "bpe" or (tokenizer_type is None and get_tokenizer_type() == "bpe"):
            embeddings_path = get_model_path("bpe_embeddings_model")
        else:
            embeddings_path = get_model_path("dynamic_embeddings_model")
        
        return load_embeddings(tokenizer_type=tokenizer_type, checkpoint_path=embeddings_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main(args=None):
    """Main function with configuration support"""
    parsed_args = parse_arguments(args)
    
    # Show configuration if requested
    if parsed_args.show_config:
        print_config()
        return
    
    mode = parsed_args.mode
    if not mode:
        usage()

    # ────────────────────────────── TRAIN ─────────────────────────────
    if mode == "train":
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        
        # Parse budget options if provided
        budget_options = None
        if parsed_args.budgets:
            budget_options = [int(b.strip()) for b in parsed_args.budgets.split(',')]
            print(f"Using custom budgets: {budget_options}")
        
        train_agent(
            "./fhe_rl/datasets/final_llm_dataset.txt",
            embeddings,
            total_timesteps=parsed_args.timesteps,
            num_envs=parsed_args.num_envs,
            budget_options=budget_options,
            denom_factor=parsed_args.denom_factor,
            constraint_method=parsed_args.method,
            budget_encoding=parsed_args.budget_encoding,
            ent_coef=parsed_args.ent_coef,
            curriculum=parsed_args.curriculum,
            algo=parsed_args.algo,
        )

    # ─────────────────────────────── TEST ─────────────────────────────
    elif mode == "test":
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)

        # Model path: CLI arg or fallback to config
        if parsed_args.model:
            agent_zip = parsed_args.model
        else:
            agent_zip = get_model_path("agent_model")

        # Budgets to test on
        test_budgets = None
        if parsed_args.budgets:
            test_budgets = [int(b.strip()) for b in parsed_args.budgets.split(',')]
            print(f"Testing on budgets: {test_budgets}")

        # Budgets the model was trained on (for obs space)
        train_budgets = None
        if parsed_args.train_budgets:
            train_budgets = [int(b.strip()) for b in parsed_args.train_budgets.split(',')]
        elif test_budgets:
            train_budgets = test_budgets

        benchmark_file = parsed_args.benchmark or "./fhe_rl/datasets/benchmarks.txt"
        test_fn = test_agent_v2 if parsed_args.test_mode == "v2" else test_agent
        test_fn(
            benchmark_file,
            embeddings,
            agent_zip,
            budget_options=train_budgets,
            test_budgets=test_budgets,
            constraint_method=parsed_args.method,
            output_file=parsed_args.output,
        )

    # ─────────────────────────────── RUN ──────────────────────────────
    elif mode == "run":
        agent_zip = get_model_path("agent_model")
        input_file = parsed_args.input_expr_file
        output_file = parsed_args.output_vector_file
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        run_agent(input_file, embeddings, agent_zip, output_file, noise_budget=300)

    else:
        print("Invalid command. Use 'train', 'test' or 'run'.")
        usage()




if __name__ == "__main__":
    main()
