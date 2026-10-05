import os
import shutil
import subprocess
import csv
import re
import statistics
from RL.fhe_rl.pareto import generate_pref_list

# Specify the parent folder containing the benchmarks and build subfolders
benchmarks_folder = "benchmarks"  
build_folder = os.path.join("build", "benchmarks")
operations = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply"]

infos = ["benchmark", "w_ops", "w_keys"]
additional_infos = [
    "Depth", "Multiplicative Depth", "compile_time (s)", "circuit_execution_time (s)",
    "galois_keys_generation_time (s)", "total_execution_time (s)", "Remaining_noise_budget",
    "rotation_keys_size (MB)", "final_ops_cost", "final_keys_cost",
]
infos.extend(operations) 
infos.extend(additional_infos) 

# Compilation initiale via CMake
try:
    print("run=> cmake', '-S', '.', '-B', 'build' ")
    result = subprocess.run(
        ['cmake', '-S', '.', '-B', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    print("run=> 'cmake', '--build', 'build'")
    result = subprocess.run(
        ['cmake', '--build', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )  
except subprocess.CalledProcessError as e:
    print(f"Command failed with error:\n{e.stderr}")    

benchmark_folders = ["max","sort","box_blur","lin_reg","hamming_dist","poly_reg","l2_distance","dot_product","gx_kernel","gy_kernel","roberts_cross","matrix_mul"] 
exceptions = ["max","sort","discrete_cosin_transform","poly_derivative"]
benchmarks_slot_counts = {
    "max" : [3,4,5], 
    "sort" : [3,4],
    "discrete_cosin_transform":[1],
    "poly_derivative":[1]
} 

# Configurations des expériences MORL
optimization_method = 1 # 0 = egraph (default), 1 = RL
cse_enabled = 1
vectorize_code = 1 
slot_counts = [3,4,5,8,16,32]
pref_list = generate_pref_list(3)  # Génère la grille de préférences de Pareto
iterations = 2 # minimum 2
window_size = 0    
depths = [5,10] 
regimes = ["50-50","100-50","100-100"]
number_instances_each_polynomial_configuration = 1
compile_time_timeout_seconds = 7200

output_csv = f"results_{'RL_MORL' if optimization_method == 1 else 'EGraph'}.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(infos)

for subfolder_name in benchmark_folders:
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name) 
    
    if os.path.isdir(build_path):
        updated_slot_counts = slot_counts 
        if subfolder_name in exceptions:
            updated_slot_counts = benchmarks_slot_counts[subfolder_name]
            
        for slot_count in updated_slot_counts:
            for w in pref_list:
                w_ops, w_keys = w
                try :
                    benchmark_compilation_timed_out = False
                    print("****************************************************************")
                    print(f"***** Run {subfolder_name} | Slots: {slot_count} | w_ops: {w_ops}, w_keys: {w_keys} *****")
                    
                    operation_stats = {
                        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
                        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
                        "compile_time (s)": [], "circuit_execution_time (s)": [], 
                        "galois_keys_generation_time (s)": [], "total_execution_time (s)": [],
                        "Remaining_noise_budget": [], "rotation_keys_size (MB)": [],
                        "final_ops_cost": [], "final_keys_cost": []
                    }
                    
                    if not subfolder_name in exceptions :
                        pro = subprocess.Popen(['python3', 'generate_{}.py'.format(subfolder_name),'--slot_count',str(slot_count)], cwd=build_path)
                        pro.wait()
                        
                    for iteration in range(iterations):
                        print(f"===> Running iteration : {iteration + 1}")
                        
                        # Signature compatible avec la gestion automatique d'arguments : framework = morl, backend = 0
                        benchmark_run_command = f"./{subfolder_name} {vectorize_code} {slot_count} morl {optimization_method} {window_size} 1 {cse_enabled} 1 0 {w_ops} {w_keys}"
                        
                        try: 
                            result = subprocess.run(
                                benchmark_run_command, shell=True, check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, 
                                cwd=build_path,
                                timeout=compile_time_timeout_seconds 
                            )
                            lines = result.stdout.splitlines()

                            compile_time_found = False 
                            for line in lines: 
                                clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).lower()
                                
                                if "final ops cost:" in clean_line:
                                    try:
                                        operation_stats["final_ops_cost"].append(float(clean_line.split(":")[1].strip()))
                                    except (IndexError, ValueError): pass
                                        
                                if "final keys cost:" in clean_line:
                                    try:
                                        operation_stats["final_keys_cost"].append(float(clean_line.split(":")[1].strip()))
                                    except (IndexError, ValueError): pass
                                        
                                if ' ms' in line and not compile_time_found:
                                    optimization_time = float(line.split()[0])
                                    operation_stats["compile_time (s)"].append(optimization_time)
                                    compile_time_found = True
                                    
                            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
                            depth = int(depth_match.group(1)) if depth_match else None
                            multiplicative_depth = int(depth_match.group(2)) if depth_match else None
                            
                            operation_stats["Depth"].append(depth)
                            operation_stats["Multiplicative Depth"].append(multiplicative_depth)
                        except subprocess.TimeoutExpired:
                            print(f"Command timed out.")
                            benchmark_compilation_timed_out = True
                        except subprocess.CalledProcessError as e:
                            print(f"Command failed: {e.stderr}")
                            continue
                            
                        if benchmark_compilation_timed_out: 
                            break 

                        # Compilation et exécution du code FHE généré
                        build_path_he = os.path.join(build_path, "he")
                        subprocess.run(['cmake', '-S', '.', '-B', 'build'], cwd=build_path_he, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        subprocess.run(['cmake', '--build', 'build'], cwd=build_path_he, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        build_path_he_build = os.path.join(build_path_he, "build")
                        
                        if iteration == iterations - 1 :
                            try:
                                for counter in range(iterations):
                                    res_run = subprocess.run(
                                        "./main", shell=True, check=True, 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                        universal_newlines=True, cwd=build_path_he_build
                                    )
                                    if counter > 0 :
                                        out_lines = res_run.stdout.splitlines()
                                        for line in out_lines:
                                            if 'circuit_execution_time_(ms):' in line:
                                                operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                                            if 'galois_keys_generation_time_(ms):' in line:
                                                operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                                            if 'total_execution_time_(ms):' in line:
                                                operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                                            if 'Remaining_noise_budget:' in line:
                                                operation_stats["Remaining_noise_budget"].append(int(line.split()[1]))
                                            if 'rotation_keys_size_(MB):' in line:
                                                operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                            except subprocess.CalledProcessError:
                                print(f"Failed in running FHE code for {subfolder_name}")
                                continue

                        # Analyse des occurrences d'opérations
                        file_name = os.path.join(build_path_he, "_gen_he_fhe.cpp")
                        if os.path.exists(file_name):
                            with open(file_name, "r") as f_code:
                                file_content = f_code.read()
                                for op in operations:
                                    nb_occurrences = len(re.findall(rf'\b{op}', file_content))
                                    operation_stats[op].append(int(nb_occurrences))
                                    
                    bench_name = f"{subfolder_name}_{slot_count}"
                    row = [bench_name, w_ops, w_keys]
                    
                    if not benchmark_compilation_timed_out : 
                        for key, values in operation_stats.items():
                            if not values:
                                result = "N/A"
                            else : 
                                result = statistics.median(values) 
                                if "time (s)" in key :
                                    result = format(result / 1000, ".3f")
                            row.append(result)
                                     
                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                except Exception as e:
                    print(f"Error for {subfolder_name}: {e}")
                    continue


######################################################################################
# Polynomials Coyote Benchmarks Block
######################################################################################
print("Run polynomial benchmarks !!!!!! ")
polynomial_folders = ["polynomials_coyote"]

for subfolder_name in polynomial_folders:
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name)
    
    for regime in regimes:
        for tree_depth in depths:
            for instance in range(1, number_instances_each_polynomial_configuration + 1):
                for w in pref_list:
                    w_ops, w_keys = w
                    try:
                        benchmark_compilation_timed_out = False
                        operation_stats = {
                            "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
                            "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
                            "compile_time (s)": [], "circuit_execution_time (s)": [],
                            "galois_keys_generation_time (s)": [], "total_execution_time (s)": [],
                            "Remaining_noise_budget": [], "rotation_keys_size (MB)": [],
                            "final_ops_cost": [], "final_keys_cost": []
                        }
                        
                        benchmark_name = f"tree_{regime}_{tree_depth}_{instance}"
                        print(f"Benchmark '{benchmark_name}' will be run...")
                        
                        for iteration in range(iterations):
                            if os.path.isdir(build_path):
                                print(f"=========> Iteration : {iteration+1}")
                                
                                # Commande polynomiale modifiée pour inclure le backend
                                command = f"./{subfolder_name} {tree_depth} {instance} {regime} {vectorize_code} {optimization_method} {window_size} 1 {cse_enabled} 1 0 {w_ops} {w_keys}"
                                
                                try:
                                    result = subprocess.run(
                                        command, shell=True, check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, cwd=build_path,
                                        timeout=compile_time_timeout_seconds,
                                    )
                                    lines = result.stdout.splitlines()
                                    compile_time_found = False
                                    
                                    for line in lines:
                                        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).lower()
                                        
                                        if "final ops cost:" in clean_line:
                                            try: operation_stats["final_ops_cost"].append(float(clean_line.split(":")[1].strip()))
                                            except: pass
                                        if "final keys cost:" in clean_line:
                                            try: operation_stats["final_keys_cost"].append(float(clean_line.split(":")[1].strip()))
                                            except: pass
                                        if " ms" in line and not compile_time_found:
                                            optimization_time = float(line.split()[0])
                                            operation_stats["compile_time (s)"].append(optimization_time)
                                            compile_time_found = True
                                            
                                    depth_match = re.search(r"max:\s*\((\d+),\s*(\d+)\)", result.stdout)
                                    depth = int(depth_match.group(1)) if depth_match else None
                                    multiplicative_depth = int(depth_match.group(2)) if depth_match else None
                                    
                                    operation_stats["Depth"].append(depth)
                                    operation_stats["Multiplicative Depth"].append(multiplicative_depth)
                                    
                                except subprocess.TimeoutExpired:
                                    print(f"Command timed out.")
                                    benchmark_compilation_timed_out = True
                                except subprocess.CalledProcessError as e:
                                    print(f"Command failed:\n{e.stderr}")
                                    
                                if benchmark_compilation_timed_out:
                                    break
                                
                                # Compilation HE
                                build_path_he = os.path.join(build_path, "he")
                                try:
                                    subprocess.run(["cmake", "-S", ".", "-B", "build"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=build_path_he)
                                    subprocess.run(["cmake", "--build", "build"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=build_path_he)
                                except:
                                    print(f"Failed in building fhe_code for benchmark:{subfolder_name}")
                                    continue
                                
                                build_path_he_build = os.path.join(build_path_he, "build")
                                
                                if iteration == iterations - 1:
                                    try:
                                        for counter in range(iterations):
                                            command = f"./main"
                                            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=build_path_he_build)
                                            if counter > 0:
                                                lines = result.stdout.splitlines()
                                                for line in lines:
                                                    if "circuit_execution_time_(ms):" in line:
                                                        operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                                                    if "galois_keys_generation_time_(ms):" in line:
                                                        operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                                                    if "total_execution_time_(ms):" in line:
                                                        operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                                                    if "rotation_keys_size_(MB):" in line:
                                                        operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                                                    if "Remaining_noise_budget:" in line:
                                                        operation_stats["Remaining_noise_budget"].append(int(line.split()[1]))
                                    except subprocess.CalledProcessError:
                                        print(f"Failed in running fhe_code for {subfolder_name}")
                                        continue

                            # Parse operation counts
                            file_name = os.path.join(build_path_he, "_gen_he_fhe.cpp")
                            if os.path.exists(file_name):
                                with open(file_name, "r") as file:
                                    file_content = file.read()
                                    for op in operations:
                                        nb_occurrences = len(re.findall(rf"\b{op}", file_content))
                                        operation_stats[op].append(int(nb_occurrences))

                        row = [benchmark_name, w_ops, w_keys]
                        if not benchmark_compilation_timed_out:
                            for key, values in operation_stats.items():
                                if not values:
                                    result = "N/A"
                                else:
                                    result = statistics.median(values)
                                    if "time (s)" in key:
                                        result = format(result / 1000, ".3f")
                                row.append(result)

                        with open(output_csv, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(row)
                    except Exception as e:
                        print(f"Command for {subfolder_name} failed with error:\n{e}")
                        continue