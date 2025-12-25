## Budget Strategy Playbook

Three policy-conditioning strategies are now available. Select them via the
`BUDGET_STRATEGY` environment variable (or `--budget_strategy` flag).

| Strategy | Env Value | Observation | When to use |
| --- | --- | --- | --- |
| Fixed one-hot | `one_hot` | `256 + 10` dims (10-slot one-hot) | Training on the curated set of budgets {200, 250, 300, 350, 400, 500, 600, 700, 1000, 9M}. |
| Threshold embedding | `budget_threshold` | `256 + 8` dims (2-dim scalar → 8-dim MLP) | Any numeric budget in `[100, 1000]` plus the unconstrained case (>1000 → “infinite”). |
| Remaining budget embedding | `remaining_budget` | `256 + 8` dims | Same as threshold strategy, but provides the remaining budget after each action. |

### Local → HPC sync

From your laptop:

**Manual file transfer (recommended):**

From your laptop, copy files individually:

```bash
# Copy Python files
cd /Users/pc/projects/chehab/Chehab-constr/chehab
scp RL/fhe_rl/config.py RL/fhe_rl/env.py RL/fhe_rl/policy.py \
    RL/fhe_rl/wrappers.py RL/fhe_rl/logger.py RL/fhe_rl/train.py \
    RL/fhe_rl/test.py RL/fhe_rl/__main__.py \
    mxa5343@jubail.abudhabi.nyu.edu:/scratch/mxa5343/chehab-constr-picked-agent/chehab/RL/fhe_rl/

# Copy SBATCH scripts
scp scripts/train_budget_*.sbatch scripts/test_budget_strategies.sbatch \
    mxa5343@jubail.abudhabi.nyu.edu:/scratch/mxa5343/chehab-constr-picked-agent/chehab/scripts/

# Copy documentation
scp docs/budget_strategies.md \
    mxa5343@jubail.abudhabi.nyu.edu:/scratch/mxa5343/chehab-constr-picked-agent/chehab/docs/
```

### Training jobs (64 GB per job)

Submit once per strategy:

```bash
cd /scratch/mxa5343/chehab-constr-picked-agent/chehab/scripts
sbatch train_budget_one_hot.sbatch
sbatch train_budget_threshold.sbatch
sbatch train_budget_remaining.sbatch
```

Each script:

* runs on a single node (`-c 16`, `--mem 64GB`, 7-day walltime),
* exports `BUDGET_STRATEGY` before invoking `python -m fhe_rl ... train`,
* streams logs to `logs/train_budget_<strategy>.log` and SLURM output files under `/scratch/.../logs`.

### Testing jobs

1. Edit `scripts/test_budget_strategies.sbatch` by adding calls such as:
   ```bash
   test_model "12800001" "budget_one_hot" "one_hot" "200,300,500,9000000"
   test_model "12800002" "budget_threshold" "budget_threshold" "200,400,800,inf"
   test_model "12800003" "budget_remaining" "remaining_budget" "300,600,inf"
   ```
2. Submit:
   ```bash
   sbatch test_budget_strategies.sbatch
   ```

The helper automatically:

* points to `eval/best_model_model_<JOB_ID>/best_model.zip`,
* sets both `BUDGET_STRATEGY` env var and the `--budget_strategy` CLI flag,
* forwards optional `--budgets` (use `inf` for the unconstrained case),
* saves each Excel sheet as `test_results_<LABEL>.xlsx`.

### Manual CLI usage

```bash
# Train with threshold embedding
BUDGET_STRATEGY=budget_threshold python -m fhe_rl --tokenizer_type dynamic train

# Test a model on multiple budgets (300, 500, infinite)
BUDGET_STRATEGY=budget_threshold python -m fhe_rl --tokenizer_type dynamic \
  test --model_path path/to/best_model.zip --budget_strategy budget_threshold \
  --budgets 300,500,inf
```

These commands also work locally; on HPC they inherit the requested 64 GB limit from the SBATCH scripts.

