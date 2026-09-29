# CHEHAB: A Fully Homomorphic Encryption Compiler

## 🧠 Overview

Fully Homomorphic Encryption (FHE) allows computations on encrypted data **without decrypting it**, enabling privacy-preserving computation. Despite its potential, FHE adoption remains limited due to challenges such as:

- High computational overhead 
- Parameter tuning complexity
- Tedious ciphertext management

**CHEHAB** addresses these challenges through a specialized FHE compiler built with the following goals:

- A **Domain-Specific Language (DSL)** for describing FHE computations.
- **Automatic parameter selection** and ciphertext maintenance.
- Automated **optimization of the user code** using multiple techniques:
  - Constant Folding (CF)
  - Common Subexpression Elimination (CSE)
  - Term Rewriting System (TRS) for complex transformations implementing advanced techniques such as reinforcement learning

Currently, CHEHAB supports the **CKKS scheme** and targets the **Lattigo backend** (v5) for homomorphic execution.

---

## 📘 DSL Example

Below is a simple example written in CHEHAB's DSL:

```cpp
#include "fheco/fheco.hpp"
using namespace fheco;

void example()
{
  Ciphertext c0("c0"); 
  Ciphertext c1 = c0 << 1;
  Ciphertext c2 = c0 << 5;
  Ciphertext c3 = c0 << 6;
  Ciphertext c4 = c1 + c0;
  Ciphertext c5 = c2 + c3;
  Ciphertext c6 = c4 + c5;
  c6.set_output("c6");
}
```

---

## ⚙️ Dev Environment Setup

### 🔧 Prerequisites

- GCC and G++ compilers
- CMake
- Go (v1.18 or higher recommended, to compile and run the Lattigo backend)
- Rust (for TRS/e-graph optimizer via `egg`)
- Python 3 (for RL optimization)

### 1. Clone the Repository

```bash
cd /scratch/<your_user_id>/
git clone https://github.com/Modern-Compilers-Lab/CHEHAB.git
cd CHEHAB
```

This directory contains an `environment.yml` for setting up the Conda environment.

### 2. Create and Activate Conda Environment

```bash
conda env create -f environment.yml -n chehabEnv
conda activate chehabEnv
cd RL/pytrs
pip3 install -e . # To install the pytrs package
```

### 3. Install Go

Ensure Go is installed on your system. If it is not installed, you can install it using the following commands (for Linux/Ubuntu):

```bash
wget https://go.dev/dl/go1.21.1.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.1.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```
*(For other operating systems, please refer to the [official Go installation guide](https://go.dev/doc/install).)*

You can verify the installation by running:
```bash
go version
```
No further installation is required at this stage; the Go modules for Lattigo will be fetched automatically when you compile the generated files.

---

## 🚀 Building and Running Benchmarks

### 🔨 Build the Compiler

```bash 
cd CHEHAB
cmake -S . -B build
cd build
make
```

### 📊 Compiler Directory Structure

- Benchmarks: `benchmarks/<benchmark_name>/`
- Equality saturation framework: `egraphs/`
- Reinforcement learning framework: `RL/`
- Core compiler: `src/`

Most benchmarks use a generator script `generate_<benchmark>.py` to:
1. Generate random input values.
2. Run reference plaintext computations.
3. Save inputs and outputs in `<benchmark>_io_example.txt`.
4. Check that the decrypted result from the homomorphic evaluation matches the plaintext python script output.

A benchmark is run in two phases:  
- **First Phase:** Triggers our compiler to translate the DSL program into a program using FHE native primitives. Here you can choose between two optimization frameworks: **e-graphs** or **RL** (Reinforcement Learning).
- **Second Phase:** Concrete homomorphic evaluation via the Lattigo backend. 

In the following steps, we use the **dot product** benchmark as an example.

### 🧪 Run the Benchmark (First Phase)

Before running the benchmark, you need to select an optimization framework. The compiler supports two advanced RL-based frameworks:
- **`morl` (Multi-Objective RL):** Optimizes execution time and rotation key size according to the user's preferences.
- **`constrained`:** Optimizes execution time with a strict noise budget.

Navigate to your benchmark folder in the `build` directory and run the compilation. 

```bash
cd build/benchmarks/dot_product
python3 generate_dot_product.py --slot_count 4
./dot_product 1 4 morl 1 0 1 1 1 1 0.8 0.2
```

*(Note: The 3rd argument `morl` selects the Multi-Objective RL framework. You can replace this with `constrained` to use the constrained optimization framework instead.)*

#### General Command Format

```bash
./<benchmark> <vectorize_code> <slot_count> <framework> <optimization_method> <window> <call_quantifier> <cse> <const_folding> <backend> <w_ops> <w_keys>
```

| Argument            | Description                               |
|---------------------|-------------------------------------------|
| `vectorize_code`    | 0/1 - Scalar or vectorized code           |
| `slot_count`        | Number of input slots                     |
| `framework`         | The optimization framework to use (e.g., `morl`, `constrained`) |
| `optimization_method` | Specific optimization mode within the framework (e.g., 0 for e-graph, 1 for RL) |
| `window`            | Vectorization window size                 |
| `call_quantifier`   | 0/1 - Enable metric collection            |
| `cse`               | 0/1 - Enable common subexpression elimination |
| `const_folding`     | 0/1 - Enable constant folding             |
| `backend`           | Backend target (0 for SEAL, 1 for Lattigo, 2 for HEonGPU). For our Lattigo guide, use `1` |
| `w_ops`             | Weight for operations in RL (e.g., 0.8)   |
| `w_keys`            | Weight for keys in RL (e.g., 0.2)         |

---

## 🔐 Final Homomorphic Execution (Second Phase)

The first phase automatically creates a Go file named `generated_fhe.go` inside the current benchmark's build folder (e.g., `build/benchmarks/dot_product`). This file contains the complete Lattigo code.

1. Ensure you are in the directory containing `generated_fhe.go`.

2. Initialize a new Go module and install the Lattigo dependency (only needed the first time you run an execution in a new folder).

**General Case:**
```bash
go mod init <benchmark_name>
go get github.com/tuneinsight/lattigo/v5/...
```

**Example (for the dot_product benchmark):**
```bash
go mod init dot_product
go get github.com/tuneinsight/lattigo/v5/...
```

3. Trigger the concrete homomorphic evaluation:

```bash
go run generated_fhe.go
```
