import os
import re
import csv
import subprocess

# ---------------------------------------------------------------------------
# Meme structure que run_morl_benchmarks_budget.py, mais au lieu de compiler
# et d'executer des projets C++ (lin_reg, box_blur, ...), on appelle
# directement `python -m fhe_rl run` (celui-la meme que le C++ appelle via
# std::system, cf. commentaire FHECO_NOISE_BUDGET dans run_morl_benchmarks_budget.py)
# sur chaque expression des datasets "benchmarks.txt" / "benchmarks extended".
#
# Run (depuis RL/, env conda/venv actif) :
#     python run_morl_benchmarks_extended.py
# Sortie : results_RL_extended.csv
# ---------------------------------------------------------------------------

# Meme sweep que run_morl_benchmarks_budget.py
pref_list = [[1.0, 0.0]]
budget_list = [300, 1000, 1000000]

# datasets a parcourir : (fichier, etiquette)
datasets = [
    ("fhe_rl/datasets/benchmarks.txt", "benchmarks"),
   
]

tmp_dir = "tmp_extended_run"
os.makedirs(tmp_dir, exist_ok=True)

header = ["dataset", "benchmark", "w_ops", "w_keys", "noise_budget",
          "final_ops_cost", "final_keys_cost", "rotation_count_final_expr",
          "compile_time (s)"]

with open("results_RL_extended.csv", mode="w", newline="") as f:
    csv.writer(f).writerow(header)


def load_named_expressions(path):
    """Meme format que benchmarks.txt : 'expr : nom' ou juste 'expr' par ligne."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                expr, name = line.rsplit(":", 1)
                out.append((expr.strip(), name.strip()))
            else:
                out.append((line, f"expr_{len(out)}"))
    return out


def run_expression(expr, name, w_ops, w_keys, noise_budget, dataset_label):
    in_path = os.path.join(tmp_dir, f"{dataset_label}_{name}_in.txt")
    out_path = os.path.join(tmp_dir, f"{dataset_label}_{name}_out.txt")

    with open(in_path, "w") as f:
        f.write(expr + "\n")

    cmd = ["python", "-m", "fhe_rl", "run", in_path, out_path,
           "--w_ops", str(w_ops), "--w_keys", str(w_keys),
           "--noise_budget", str(noise_budget)]

    final_ops_cost = "N/A"
    final_keys_cost = "N/A"
    rotation_count = "N/A"
    compile_time = "N/A"

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=1800)

        if res.returncode != 0:
            print(f"\n[CRITICAL ERROR] {dataset_label}:{name} crashed!")
            print(f"--- stderr ---\n{res.stderr.strip()}\n--------------")

        for line in res.stdout.splitlines():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

            ops_match = re.search(r'final\s*exec\s*cost\s*:\s*([\d\.]+)', clean, re.IGNORECASE)
            if ops_match:
                final_ops_cost = float(ops_match.group(1))

            keys_match = re.search(r'final\s*keys\s*cost\s*:\s*([\d\.]+)', clean, re.IGNORECASE)
            if keys_match:
                final_keys_cost = float(keys_match.group(1))

            time_match = re.search(r'completed in\s*([\d\.]+)\s*seconds', clean, re.IGNORECASE)
            if time_match:
                compile_time = float(time_match.group(1))

        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                final_expr_line = f.readline()
            rotation_count = final_expr_line.count("<<")

    except Exception as e:
        print(f"Python exception while running {dataset_label}:{name}: {e}")

    return [dataset_label, name, w_ops, w_keys, noise_budget,
            final_ops_cost, final_keys_cost, rotation_count, compile_time]


for dataset_path, dataset_label in datasets:
    if not os.path.exists(dataset_path):
        print(f"[skip] dataset introuvable : {dataset_path}")
        continue

    expressions = load_named_expressions(dataset_path)

    for expr, name in expressions:
        for w_ops, w_keys in pref_list:
            for budget in budget_list:
                print(f"*****run {dataset_label}:{name} , w=({w_ops},{w_keys}) , budget={budget}******")
                row = run_expression(expr, name, w_ops, w_keys, budget, dataset_label)

                with open("results_RL_extended.csv", mode="a", newline="") as f:
                    csv.writer(f).writerow(row)

                if isinstance(row[7], int) and row[7] > 0:
                    print(f"    -> ROTATION dans la formule finale ({row[7]}x) !")