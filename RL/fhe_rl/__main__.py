import sys
import os
import argparse

# Support framework flag (--framework constrained/morl)
framework_parser = argparse.ArgumentParser(add_help=False)
framework_parser.add_argument("--framework", choices=["constrained", "morl"], default="constrained", help='Framework to use (default: constrained)')
args, _ = framework_parser.parse_known_args()
try:
    import pytrs.config
    pytrs.config.framework = args.framework
except ImportError:
    pass

from .run import run_agent
from .train import train_agent
from .train_mo import train_agent_mo
from .test import test_agent, test_agent_v2
from .utils import load_embeddings_from_config
from .TRAE_bpe import BPETokenizer  # Import for pickle compatibility
from .morl import run_interactive, add_subparser
from .config import (
    get_model_path, get_tokenizer_type, 
    print_config, set_framework
)


def parse_arguments(args=None):
    """Parse command line arguments"""
    global framework_parser
    parser = argparse.ArgumentParser(description="FHE RL Agent", parents=[framework_parser])
    
    # Tokenizer type selection
    parser.add_argument(
        '--tokenizer_type', 
        choices=['dynamic', 'bpe'], 
        default=get_tokenizer_type(),
        help='Tokenizer type to use (default: from config)'
    )
    
    parser.add_argument(
        '--show_config', 
        action='store_true',
        help='Show current configuration and exit'
    )
    
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
        help='Comma-separated list of noise budgets (e.g., "230,369,9000")'
    )
    train_parser.add_argument(
        '--total_timesteps', '--timesteps',
        dest='total_timesteps',
        type=int,
        default=2_000_000,
        help='Total training timesteps (default: 2000000)'
    )
    train_parser.add_argument(
        '--eval_freq',
        type=int,
        default=10000,
        help='Evaluation frequency for MORL (default: 10000)'
    )
    train_parser.add_argument(
        '--n_envs', '--num_envs',
        dest='n_envs',
        type=int,
        default=8,
        help='Number of parallel environments (default: 8)'
    )
    train_parser.add_argument(
        '--denom_factor',
        type=int,
        default=4,
        help='Lagrangian update frequency denominator (default: 4)'
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
    train_parser.add_argument(
        '--curriculum',
        action='store_true',
        default=False,
        help='Enable curriculum budget scheduling'
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
        help='film_a vs film_b policy comparison alternative'
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
        help='Comma-separated budgets to TEST on'
    )
    test_parser.add_argument(
        '--train_budgets',
        type=str,
        default=None,
        help='Comma-separated budgets the model was TRAINED on'
    )
    test_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_pid',
        choices=['none', 'lagrangian_od_ov', 'lagrangian_perstep', 'lagrangian_always_done', 'margin_barrier', 'noise_masking', 'nato_sc', 'lagrangian_pid'],
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
        help='Path to benchmark expressions file'
    )
    test_parser.add_argument(
        '--save_optimized',
        type=str,
        default=None,
        help='Save RL-optimized expressions to this file'
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
        default=None,
        help='Noise budget for execution'
    )
    run_parser.add_argument(
        '--train_budgets',
        type=str,
        default=None,
        help='Comma-separated budgets the model was TRAINED on'
    )
    run_parser.add_argument(
        '--method',
        type=str,
        default='lagrangian_pid',
        help='Constraint method the model was trained with'
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
        "  python -m fhe_rl --show_config  # Show current configuration\n"
    )
    sys.exit(1)


def main(args=None):
    """Main function with configuration support"""
    parsed_args = parse_arguments(args)
    
    # Configure framework (Imed / MORL)
    framework_name = "mo" if parsed_args.framework == "morl" else parsed_args.framework
    set_framework(framework_name)
    
    if parsed_args.show_config:
        print_config()
        return
    
    mode = parsed_args.mode
    if not mode:
        usage()

    # ────────────────────────────── TRAIN ─────────────────────────────
    if mode == "train":
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        
        budget_options = None
        if parsed_args.budgets:
            budget_options = [int(b.strip()) for b in parsed_args.budgets.split(',')]
            print(f"Using custom budgets: {budget_options}")
        
        if framework_name == "mo":
            train_agent_mo(
                parsed_args.dataset,
                embeddings,
                total_timesteps=parsed_args.total_timesteps,
                num_envs=parsed_args.n_envs,
                ent_coef=parsed_args.ent_coef,
                n_cycle=parsed_args.n_cycle,
                n_budget=parsed_args.n_budget,
                lambda_env=parsed_args.lambda_env,
                lambda_kl=parsed_args.lambda_kl,
                eval_freq=parsed_args.eval_freq,
            )
        else:
            train_agent(
                parsed_args.dataset,
                embeddings,
                total_timesteps=parsed_args.total_timesteps,
                num_envs=parsed_args.n_envs,
                budget_options=budget_options,
                denom_factor=parsed_args.denom_factor,
                constraint_method=parsed_args.method,
                budget_encoding=parsed_args.budget_encoding,
                ent_coef=parsed_args.ent_coef,
                curriculum=parsed_args.curriculum,
                algo=parsed_args.algo,
                n_cycle=parsed_args.n_cycle,
                n_budget=parsed_args.n_budget,
                lambda_env=parsed_args.lambda_env,
                lambda_kl=parsed_args.lambda_kl,
                policy_variant=parsed_args.policy_variant,
            )

    # ─────────────────────────────── TEST ─────────────────────────────
    elif mode == "test":
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)

        agent_zip = parsed_args.model if parsed_args.model else get_model_path("agent_model")

        test_budgets = None
        if parsed_args.budgets:
            test_budgets = [int(b.strip()) for b in parsed_args.budgets.split(',')]

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
        if framework_name == "mo":
            agent_zip = get_model_path("mo_agent_model")
            from .run_mo import run_agent_mo
            embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
            run_agent_mo(parsed_args.input_expr_file, embeddings, agent_zip, parsed_args.output_vector_file, w_ops=parsed_args.w_ops, w_keys=parsed_args.w_keys)
        else:
            agent_zip = get_model_path("agent_model")
            embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
            
            noise_budget = parsed_args.noise_budget
            if noise_budget is None:
                env_nb = os.environ.get('FHECO_NOISE_BUDGET')
                noise_budget = int(env_nb) if env_nb is not None else 300

            train_budgets = None
            if parsed_args.train_budgets:
                train_budgets = [int(b.strip()) for b in parsed_args.train_budgets.split(',')]

            run_agent(parsed_args.input_expr_file, embeddings, agent_zip, parsed_args.output_vector_file,
                      noise_budget=noise_budget,
                      w_ops=parsed_args.w_ops, w_keys=parsed_args.w_keys,
                      budget_options=train_budgets,
                      constraint_method=parsed_args.method)

    # ────────────────────────── INTERACTIVE ───────────────────────────
    elif mode == "interactive":
        run_interactive(mode=getattr(parsed_args, "interactive_mode", None))    
        
    else:
        print("Invalid command. Use 'train', 'test', 'run', or 'interactive'.")
        usage()


if __name__ == "__main__":
    main()