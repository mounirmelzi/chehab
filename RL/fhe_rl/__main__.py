import sys
import os
import argparse
from .run import run_agent
from .train import train_agent
from .test import test_agent, test_agent_v2
from .utils import load_embeddings_from_config
from .config import get_model_path, print_config
from .morl import run_interactive, add_subparser

def parse_arguments(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FHE RL Agent")
    
    # Add config flag
    parser.add_argument(
        '--show_config', 
        action='store_true',
        help='Show current configuration and exit'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Available commands')
    
    # ─── TRAIN COMMAND ────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument(
        '--dataset',
        type=str,
        default='./fhe_rl/datasets/final_llm_dataset.txt',
        help='Path to the training expressions file'
    )
    train_parser.add_argument(
        '--budgets',
        type=str,
        default=None,
        help='Comma-separated list of noise budgets (e.g., "240,300,1000,9000000")'
    )
    train_parser.add_argument(
        '--total_timesteps',
        type=int,
        default=2_000_000,
        help='Total training timesteps (default: 2000000)'
    )
    train_parser.add_argument(
        '--n_envs',
        type=int,
        default=8,
        help='Number of parallel environments (default: 8)'
    )
    train_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_pid',
        choices=['none', 'lagrangian_od_ov', 'lagrangian_perstep', 'lagrangian_always_done', 'margin_barrier', 'noise_masking', 'nato_sc', 'lagrangian_pid'],
        help='Constraint enforcement method (default: lagrangian_pid)'
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
        default='film',
        choices=['raw', 'embed', 'film', 'none'],
        help='Budget encoding in policy (default: film)'
    )
    train_parser.add_argument(
        '--ent_coef',
        type=float,
        default=0.01,
        help='Entropy coefficient for exploration (default: 0.01)'
    )
    
    # MORL hyperparameters
    train_parser.add_argument(
        '--n_cycle',
        type=int,
        default=1,
        help='Biased-preference cycle length. 0 for always random. (default: 1)'
    )
    train_parser.add_argument(
        '--n_budget',
        type=int,
        default=5,
        help='Fixed normalization budget for rotation keys (default: 5)'
    )
    train_parser.add_argument(
        '--lambda_env',
        type=float,
        default=0.0,
        help='Pareto-envelope bonus weight (default: 0.0)'
    )
    train_parser.add_argument(
        '--lambda_kl',
        type=float,
        default=0.0,
        help='KL-divergence bonus weight (default: 0.0)'
    )
    train_parser.add_argument(
        '--policy_variant',
        type=str,
        default='film_a',
        choices=['film_a', 'film_b'],
        help='film_a: FiLM(budget) then concat preference after (default, current repo design). '
             'film_b: single FiLM on concat[budget, preference] (A/B comparison alternative).'
    )
    
    # ─── TEST COMMAND ─────────────────────────────────────────────────────────
    test_parser = subparsers.add_parser('test', help='Test the agent')
    test_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model .zip file'
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
        help='Comma-separated budgets the model was TRAINED on (for correct obs space)'
    )
    test_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_pid',
        help='Constraint method the model was trained with (default: lagrangian_pid)'
    )
    test_parser.add_argument(
        '--test_mode',
        type=str,
        default='v2',
        choices=['v1', 'v2'],
        help='v1=existing test, v2=trajectory checkpointing + safety rollback'
    )
    test_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output Excel file path'
    )
    test_parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        help='Path to benchmark expressions file (default: ./fhe_rl/datasets/benchmarks.txt)'
    )
    test_parser.add_argument(
        '--save_optimized',
        type=str,
        default=None,
        help='Save RL-optimized expressions to this file (one per line, expr:name format)'
    )
    
    # ─── RUN COMMAND ──────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser('run', help='Run the agent')
    run_parser.add_argument('input_expr_file', help='Input expression file')
    run_parser.add_argument('output_vector_file', help='Output vector file')
    run_parser.add_argument('--w_ops', type=float, default=1.0, help='Weight for execution time/operations')
    run_parser.add_argument('--w_keys', type=float, default=0.0, help='Weight for rotation keys')
    run_parser.add_argument(
        '--noise_budget',
        type=int,
        default=int(os.environ.get('FHECO_NOISE_BUDGET', 300)),
        help='Noise budget for execution (env var FHECO_NOISE_BUDGET overrides default 300)'
    )

    # ─── INTERACTIVE COMMAND ──────────────────────────────────────────────────
    add_subparser(subparsers)

    return parser.parse_args(args)

def usage() -> None:
    print(
        "Usage:\n"
        "  python -m fhe_rl train [options]         Train the MORL agent\n"
        "  python -m fhe_rl test [options]          Test safety rollback on benchmarks\n"
        "  python -m fhe_rl run [options] <in> <out> Run direct optimization\n"
        "  python -m fhe_rl interactive [--mode {direct,menu}]\n"
        "                                           Optimise FHE circuits interactively\n"
        "                                           direct: single preference -> one circuit\n"
        "                                           menu:   preference range  -> Pareto frontier\n"
        "\n"
        "All model paths are loaded from config.py."
    )
    sys.exit(1)

def main(args=None):
    """Main function with configuration support"""
    parsed_args = parse_arguments(args)
    
    if parsed_args.show_config:
        print_config()
        return
    
    mode = parsed_args.mode
    if not mode:
        usage()

    # ────────────────────────────── TRAIN ─────────────────────────────
    if mode == "train":
        embeddings, _ = load_embeddings_from_config()
        
        budget_options = None
        if parsed_args.budgets:
            budget_options = [int(b.strip()) for b in parsed_args.budgets.split(',')]
            print(f"Using custom budgets: {budget_options}")
        
        train_agent(
            parsed_args.dataset,
            embeddings,
            total_timesteps=parsed_args.total_timesteps,
            num_envs=parsed_args.n_envs,
            budget_options=budget_options,
            constraint_method=parsed_args.method,
            budget_encoding=parsed_args.budget_encoding,
            ent_coef=parsed_args.ent_coef,
            algo=parsed_args.algo,
            n_cycle=parsed_args.n_cycle,
            n_budget=parsed_args.n_budget,
            lambda_env=parsed_args.lambda_env,
            lambda_kl=parsed_args.lambda_kl,
            policy_variant=parsed_args.policy_variant,
        )

    # ─────────────────────────────── TEST ─────────────────────────────
    elif mode == "test":
        embeddings, _ = load_embeddings_from_config()

        agent_zip = parsed_args.model if parsed_args.model else get_model_path("agent_model")

        test_budgets = None
        if parsed_args.budgets:
            test_budgets = [int(b.strip()) for b in parsed_args.budgets.split(',')]
            print(f"Testing on budgets: {test_budgets}")

        train_budgets = None
        if parsed_args.train_budgets:
            train_budgets = [int(b.strip()) for b in parsed_args.train_budgets.split(',')]
        elif test_budgets:
            train_budgets = test_budgets

        benchmark_file = parsed_args.benchmark or "./fhe_rl/datasets/benchmarks.txt"
        test_fn = test_agent_v2 if parsed_args.test_mode == "v2" else test_agent
        
        kwargs = dict(
            budget_options=train_budgets,
            test_budgets=test_budgets,
            constraint_method=parsed_args.method,
            output_file=parsed_args.output,
        )
        if parsed_args.test_mode == "v2" and parsed_args.save_optimized:
            kwargs["save_optimized"] = parsed_args.save_optimized
            
        test_fn(benchmark_file, embeddings, agent_zip, **kwargs)

    # ─────────────────────────────── RUN ──────────────────────────────
    elif mode == "run":
        agent_zip = get_model_path("agent_model")
        input_file = parsed_args.input_expr_file
        output_file = parsed_args.output_vector_file
        embeddings, _ = load_embeddings_from_config()
        
        run_agent(input_file, embeddings, agent_zip, output_file, 
                  noise_budget=parsed_args.noise_budget, 
                  w_ops=parsed_args.w_ops, w_keys=parsed_args.w_keys)

    # ────────────────────────── INTERACTIVE ───────────────────────────
    elif mode == "interactive":
        run_interactive(mode=getattr(parsed_args, "interactive_mode", None))    
        
    else:
        print("Invalid command. Use 'train', 'test', 'run', or 'interactive'.")
        usage()

if __name__ == "__main__":
    main()