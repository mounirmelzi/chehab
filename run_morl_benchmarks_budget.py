import os
import subprocess
import csv
import re
import statistics

benchmarks_folder = "benchmarks"
build_folder = os.path.join("build", "benchmarks")
operations = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply"]
infos = ["benchmark", "w_ops", "w_keys", "noise_budget"]
additional_infos = [
    "Depth", "Multiplicative Depth", "compile_time (s)",
    "circuit_execution_time (s)", 'galois_keys_generation_time (s)', 'total_execution_time (s)',
    "Remaining_noise_budget", 'rotation_keys_size (MB)', 'rotation_keys_count', 'final_ops_cost', 'final_keys_cost'
]
infos.extend(operations)
infos.extend(additional_infos)

benchmark_folders = [
  "dot_product"
]
#  "lin_reg", "box_blur", "matrix_mul", "max", "sort", "l2_distance",
    #"poly_reg", "roberts_cross", 
pref_list = [[0.8, 0.2], [1.0, 0.0]]  # Points de grille grossière pour la bisection
budget_list = [230, 369, 9000]

depths = [5, 10]
regimes = ["50-50", "100-50", "100-100"]
number_instances_each_polynomial_configuration = 2

exceptions = ["max", "sort", "discrete_cosin_transform", "poly_derivative"]
benchmarks_slot_counts = {"max": [3, 4, 5], "sort": [3], "discrete_cosin_transform": [1], "poly_derivative": [1]}
slot_counts = [4, 8, 16, 32]

BASE_UNIT = 18
MAX_BISECT_DEPTH = 3

try:
    print("run=> cmake -S . -B build")
    subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    print("run=> cmake --build build")
    subprocess.run(['cmake', '--build', 'build'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
except subprocess.CalledProcessError as e:
    print(f"CMake build failed:\n{e.stderr}")
    raise

output_csv = "results_RL_budget.csv"
with open(output_csv, mode='w', newline='') as file:
    csv.writer(file).writerow(infos)


def run_benchmark(subfolder_name, slot_count, w_ops, w_keys, noise_budget, build_path):
    he_path = os.path.join(build_path, "he")
    build_he_path = os.path.join(he_path, "build")
    stats = {k: [] for k in operations + infos[4:]}

    if subfolder_name not in exceptions:
        gen_script = os.path.join(build_path, f"generate_{subfolder_name}.py")
        if os.path.exists(gen_script):
            subprocess.Popen(['python3', gen_script, '--slot_count', str(slot_count)], cwd=build_path).wait()

    cmd = f"./{subfolder_name} 1 {slot_count} morl 1 0 1 1 1 0 {w_ops} {w_keys}"
    run_env = os.environ.copy()
    run_env["FHECO_NOISE_BUDGET"] = str(noise_budget)

    benchmark_compilation_timed_out = False
    try:
        res = subprocess.run(cmd, shell=True, check=False,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, cwd=build_path, timeout=7200, env=run_env)

        if res.returncode != 0:
            print(f"\n[CRITICAL ERROR] benchmark {subfolder_name} crashed!")
            print(f"--- stderr ---\n{res.stderr.strip()}\n--------------")

        for line in res.stdout.splitlines():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            clean_lower = clean.lower()

            if 'final exec cost' in clean_lower or 'final ops cost' in clean_lower:
                try:
                    val_str = clean_lower.split(':')[1].strip()
                    stats["final_ops_cost"].append(float(val_str))
                except (IndexError, ValueError):
                    pass

            if 'final keys cost' in clean_lower:
                try:
                    val_str = clean_lower.split(':')[1].strip()
                    stats["final_keys_cost"].append(float(val_str))
                except (IndexError, ValueError):
                    pass

            if ' ms' in line:
                try:
                    stats["compile_time (s)"].append(float(line.split()[0]))
                except ValueError:
                    pass

        depth_m = re.search(r'max:\s*\((\d+),\s*(\d+)\)', res.stdout)
        if depth_m:
            stats["Depth"].append(int(depth_m.group(1)))
            stats["Multiplicative Depth"].append(int(depth_m.group(2)))

        if res.returncode == 0:
            subprocess.run(['cmake', '-S', '.', '-B', 'build'], cwd=he_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['cmake', '--build', 'build'], cwd=he_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            run_res = subprocess.run("./main", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     universal_newlines=True, cwd=build_he_path, env=run_env)
            for line in run_res.stdout.splitlines():
                if 'circuit_execution_time_(ms):' in line: stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                if 'galois_keys_generation_time_(ms):' in line: stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                if 'total_execution_time_(ms):' in line: stats["total_execution_time (s)"].append(float(line.split()[1]))
                if 'rotation_keys_size_(MB):' in line: stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                if 'rotation_keys_count_:' in line: stats["rotation_keys_count"].append(int(line.split()[1]))
                if 'Remaining_noise_budget:' in line: stats["Remaining_noise_budget"].append(int(line.split()[1]))

        cpp_file = os.path.join(he_path, "_gen_he_fhe.cpp")
        if os.path.exists(cpp_file):
            with open(cpp_file, "r") as f:
                content = f.read()
                for op in operations:
                    stats[op].append(len(re.findall(rf'\b{op}', content)))
    except subprocess.TimeoutExpired:
        print(f"Command timed out.")
        benchmark_compilation_timed_out = True
    except Exception as e:
        print(f"Python exception while running {subfolder_name}: {e}")
        benchmark_compilation_timed_out = True

    bench_name = f"{subfolder_name}_{slot_count}"
    row = [bench_name, w_ops, w_keys, noise_budget]
    raw_key_size = None

    if not benchmark_compilation_timed_out:
        for key in operations + infos[4:]:
            vals = stats[key]
            if not vals or None in vals:
                row.append("N/A")
            else:
                v = statistics.median(vals)
                if key in ["compile_time (s)", "circuit_execution_time (s)", "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    v = format(v / 1000, ".3f")
                if key == "final_keys_cost":
                    raw_key_size = statistics.median(vals)
                row.append(v)

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if benchmark_compilation_timed_out:
        return None, None

    key_size_level = int(raw_key_size) if raw_key_size is not None else None
    return row, key_size_level


def _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                       noise_budget, build_path, target, found, depth=0):
    if depth >= MAX_BISECT_DEPTH:
        return

    w_mid = round((w_hi + w_lo) / 2, 10)
    w_mid_keys = round(1.0 - w_mid, 10)

    row, ks_mid = run_benchmark(subfolder_name, slot_count, w_mid, w_mid_keys, noise_budget, build_path)
    if row is None or ks_mid is None:
        return

    found.add(ks_mid)

    if ks_mid == target:
        return
    elif ks_mid < target:
        _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_mid, ks_mid,
                           noise_budget, build_path, target, found, depth=depth + 1)
    else:
        _bisect_one_target(subfolder_name, slot_count, w_mid, ks_mid, w_lo, ks_lo,
                           noise_budget, build_path, target, found, depth=depth + 1)


def bisect_search(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo, noise_budget, build_path):
    lo, hi = min(ks_lo, ks_hi), max(ks_lo, ks_hi)
    missing_targets = list(range(lo + 1, hi))
    if not missing_targets:
        return

    found = set()
    for target in missing_targets:
        if target in found:
            continue
        _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                           noise_budget, build_path, target, found)


# ── Boucle principale des benchmarks standards ────────────────────────────────
for sub in benchmark_folders:
    b_path = os.path.join(build_folder, sub)
    if os.path.isdir(b_path):
        for sc in benchmarks_slot_counts.get(sub, slot_counts):
            for budget in budget_list:
                known_points = []
                for w_ops, w_keys in pref_list:
                    print(f"*****run {sub} , slot : {sc} , w=({w_ops},{w_keys}) , budget={budget}******")
                    try:
                        row, ks_level = run_benchmark(sub, sc, w_ops, w_keys, budget, b_path)
                        if ks_level is not None:
                            known_points.append((w_ops, ks_level))
                    except Exception as e:
                        print(f"Command failed: {e}")

                known_points.sort(key=lambda x: x[0])
                for i in range(len(known_points) - 1):
                    w_lo, ks_lo = known_points[i]
                    w_hi, ks_hi = known_points[i + 1]
                    if ks_lo != ks_hi:
                        try:
                            bisect_search(sub, sc, w_hi, ks_hi, w_lo, ks_lo, budget, b_path)
                        except Exception as e:
                            print(f"Bisect failed: {e}")


# ── Benchmarks polynomiaux (Polynomials Coyote) ───────────────────────────────
print("Run polynomial benchmarks !!!!!!")
polynomial_folders = ["polynomials_coyote"]

def run_poly_benchmark(subfolder_name, build_path, benchmark_name, tree_depth, instance, regime, w_ops, w_keys, noise_budget):
    build_path_he = os.path.join(build_path, "he")
    build_path_he_build = os.path.join(build_path_he, "build")
    operation_stats = {
        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
        "compile_time (s)": [], "circuit_execution_time (s)": [],
        "galois_keys_generation_time (s)": [], "total_execution_time (s)": [],
        "Remaining_noise_budget": [], "rotation_keys_size (MB)": [],
        "final_ops_cost": [], "final_keys_cost": []
    }
    benchmark_compilation_timed_out = False

    command = (f"./{subfolder_name} {tree_depth} {instance} {regime} "
               f"1 1 0 1 1 1 0 {w_ops} {w_keys}")
    run_env = os.environ.copy()
    run_env["FHECO_NOISE_BUDGET"] = str(noise_budget)

    try:
        result = subprocess.run(
            command, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, cwd=build_path,
            timeout=7200, env=run_env
        )
        for line in result.stdout.splitlines():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip().lower()
            if 'final exec cost' in clean or 'final ops cost' in clean:
                try:
                    operation_stats["final_ops_cost"].append(float(clean.split(':')[1].strip()))
                except:
                    pass
            if 'final keys cost' in clean:
                try:
                    operation_stats["final_keys_cost"].append(float(clean.split(':')[1].strip()))
                except:
                    pass
            if ' ms' in line:
                try:
                    operation_stats["compile_time (s)"].append(float(line.split()[0]))
                except:
                    pass

        depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
        if depth_match:
            operation_stats["Depth"].append(int(depth_match.group(1)))
            operation_stats["Multiplicative Depth"].append(int(depth_match.group(2)))

        if result.returncode == 0:
            subprocess.run(['cmake', '-S', '.', '-B', 'build'], cwd=build_path_he, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['cmake', '--build', 'build'], cwd=build_path_he, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            run_res = subprocess.run("./main", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     universal_newlines=True, cwd=build_path_he_build, env=run_env)
            for line in run_res.stdout.splitlines():
                if 'circuit_execution_time_(ms):' in line: operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                if 'galois_keys_generation_time_(ms):' in line: operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                if 'total_execution_time_(ms):' in line: operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                if 'rotation_keys_size_(MB):' in line: operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                if 'rotation_keys_count_:' in line: operation_stats["rotation_keys_count"].append(int(line.split()[1]))
                if 'Remaining_noise_budget:' in line: operation_stats["Remaining_noise_budget"].append(int(line.split()[1]))

        cpp_file = os.path.join(build_path_he, "_gen_he_fhe.cpp")
        if os.path.exists(cpp_file):
            with open(cpp_file, "r") as f:
                content = f.read()
                for op in operations:
                    operation_stats[op].append(len(re.findall(rf'\b{op}', content)))
    except Exception as e:
        print(f"Error in poly benchmark: {e}")
        benchmark_compilation_timed_out = True

    row = [benchmark_name, w_ops, w_keys, noise_budget]
    raw_ops_cost, raw_keys_cost = None, None

    if not benchmark_compilation_timed_out:
        for key in operations + infos[4:]:
            vals = operation_stats[key]
            if not vals or None in vals:
                row.append("N/A")
            else:
                v = statistics.median(vals)
                if key in ["compile_time (s)", "circuit_execution_time (s)", "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    v = format(v / 1000, ".3f")
                if key == "final_ops_cost":
                    raw_ops_cost = statistics.median(vals)
                if key == "final_keys_cost":
                    raw_keys_cost = statistics.median(vals)
                row.append(v)

    with open(output_csv, mode='a', newline='') as f:
        csv.writer(f).writerow(row)

    if benchmark_compilation_timed_out:
        return None, None, None

    return row, (int(raw_ops_cost) if raw_ops_cost is not None else None), (int(raw_keys_cost) if raw_keys_cost is not None else None)


def _poly_bisect_one_target(subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
                             w_hi, ks_ops_hi, ks_keys_hi,
                             w_lo, ks_ops_lo, ks_keys_lo,
                             noise_budget, target_ops, target_keys, found, depth=0):
    if depth >= MAX_BISECT_DEPTH:
        return

    w_mid = round((w_hi + w_lo) / 2, 10)
    w_mid_keys = round(1.0 - w_mid, 10)

    row, ops_mid, keys_mid = run_poly_benchmark(
        subfolder_name, build_path, benchmark_name, tree_depth, instance, regime, w_mid, w_mid_keys, noise_budget
    )
    if row is None or ops_mid is None:
        return

    found.add((ops_mid, keys_mid))

    if ops_mid == target_ops and keys_mid == target_keys:
        return

    if ops_mid < target_ops:
        _poly_bisect_one_target(
            subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
            w_hi, ks_ops_hi, ks_keys_hi, w_mid, ops_mid, keys_mid,
            noise_budget, target_ops, target_keys, found, depth=depth + 1
        )
    else:
        _poly_bisect_one_target(
            subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
            w_mid, ops_mid, keys_mid, w_lo, ks_ops_lo, ks_keys_lo,
            noise_budget, target_ops, target_keys, found, depth=depth + 1
        )


def poly_bisect_search(subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
                       w_hi, ks_ops_hi, ks_keys_hi,
                       w_lo, ks_ops_lo, ks_keys_lo, noise_budget):
    ops_lo,  ops_hi  = min(ks_ops_lo,  ks_ops_hi),  max(ks_ops_lo,  ks_ops_hi)
    keys_lo, keys_hi = min(ks_keys_lo, ks_keys_hi), max(ks_keys_lo, ks_keys_hi)

    missing_targets = [
        (o, k)
        for o in range(ops_lo + 1, ops_hi)
        for k in range(keys_lo + 1, keys_hi)
    ]

    if not missing_targets:
        return

    found = set()
    for target_ops, target_keys in missing_targets:
        if (target_ops, target_keys) in found:
            continue
        _poly_bisect_one_target(
            subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
            w_hi, ks_ops_hi, ks_keys_hi, w_lo, ks_ops_lo, ks_keys_lo,
            noise_budget, target_ops, target_keys, found
        )

for subfolder_name in polynomial_folders:
    build_path = os.path.join(build_folder, subfolder_name)
    if os.path.isdir(build_path):
        for regime in regimes:
            for tree_depth in depths:
                for instance in range(1, number_instances_each_polynomial_configuration + 1):
                    for budget in budget_list:
                        benchmark_name = f'tree_{regime}_{tree_depth}_{instance}'
                        print(f"\nBenchmark '{benchmark_name}' (budget={budget}) will be run...")

                        known_points = []
                        written_w_ops_poly = set()

                        for w_ops, w_keys in pref_list:
                            if w_ops in written_w_ops_poly:
                                continue
                            print("*" * 64)
                            try:
                                row, ops_level, keys_level = run_poly_benchmark(
                                    subfolder_name, build_path, benchmark_name, tree_depth, instance, regime, w_ops, w_keys, budget
                                )
                                written_w_ops_poly.add(w_ops)
                                if ops_level is not None and keys_level is not None:
                                    known_points.append((w_ops, ops_level, keys_level))
                            except Exception as e:
                                print(f"Command for {benchmark_name} failed with error:\n{e}")
                                continue

                        known_points.sort(key=lambda x: x[0])
                        for i in range(len(known_points) - 1):
                            w_lo, ops_lo, keys_lo = known_points[i]
                            w_hi, ops_hi, keys_hi = known_points[i + 1]
                            if ops_lo != ops_hi or keys_lo != keys_hi:
                                try:
                                    poly_bisect_search(
                                        subfolder_name, build_path, benchmark_name, tree_depth, instance, regime,
                                        w_hi, ops_hi, keys_hi, w_lo, ops_lo, keys_lo, budget
                                    )
                                except Exception as e:
                                    print(f"Poly bisect failed for {benchmark_name}: {e}")