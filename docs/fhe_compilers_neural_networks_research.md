# FHE Compilers for Neural Networks: State-of-the-Art Research Report

**Objective:** Analyze the landscape of FHE compilers that optimize and execute neural network inference on encrypted data, focusing on their optimization techniques, performance, and underlying libraries. Identify directions for extending Chehab to support NN workloads.

---

## Table of Contents

1. [Background: Why FHE + Neural Networks Is Hard](#1-background)
2. [Major FHE Compilers for Neural Networks](#2-major-fhe-compilers)
3. [Optimization Techniques](#3-optimization-techniques)
4. [Performance Benchmarks](#4-performance-benchmarks)
5. [Best-in-Class Methods per NN Operation](#5-best-methods-per-operation) **(Practical Cookbook)**
6. [Term Rewriting Approaches in FHE](#6-term-rewriting-in-fhe)
7. [Open Challenges](#7-open-challenges)
8. [Implications for Chehab](#8-implications-for-chehab)
9. [References](#9-references)

---

## 1. Background: Why FHE + Neural Networks Is Hard

FHE schemes support only two native operations: addition and multiplication over ciphertexts. Neural networks, however, require:

- **Linear layers**: matrix-vector multiplications, convolutions (feasible but expensive in FHE)
- **Non-linear activations**: ReLU, sigmoid, softmax (impossible natively; must be approximated)
- **Data movement**: pooling, batch normalization, reshaping (require costly ciphertext rotations)

The fundamental bottleneck is **multiplicative depth** — each multiplication consumes noise budget, and deep networks chain many multiplications. Without bootstrapping (noise refresh), the computation depth is bounded by the encryption parameters.

> *"Word-wise schemes (BGV, BFV, CKKS) efficiently handle linear operations like matrix multiplications but struggle with non-linear operations, while bit-wise schemes (FHEW, TFHE) handle non-linear operations well but incur prohibitively high costs on linear ones. Neural networks alternate between both, making them difficult to support efficiently."*
>
> — Lee et al., "SoK: Can Fully Homomorphic Encryption Support General AI Computation?" [[1]](#ref1)

---

## 2. Major FHE Compilers for Neural Networks

### 2.1 CryptoNets (Microsoft, 2016)

The pioneering work that demonstrated neural network inference on encrypted data. CryptoNets applied a trained CNN to MNIST data encrypted with the SEAL library, using square activation functions instead of ReLU.

- **Scheme**: Leveled HE (SEAL)
- **Performance**: 99% accuracy on MNIST, ~59,000 predictions/hour on a single PC
- **Key limitation**: Only shallow networks (2 layers), no bootstrapping

> Gilad-Bachrach et al., "CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy," ICML 2016 [[2]](#ref2)

### 2.2 CHET (Microsoft Research, 2019)

A domain-specific optimizing compiler that translates tensor programs into FHE circuits. CHET introduced the concept of a **Homomorphic Instruction Set Architecture (HISA)** — an intermediate representation that abstracts over different FHE schemes.

- **Scheme**: HEAAN (CKKS)
- **Optimizations**: Automatic encryption parameter selection, tensor layout optimization, scheme-specific transformations
- **Performance**: Evaluated SqueezeNet — at the time the deepest network run homomorphically. Generated circuits that outperformed hand-tuned expert implementations
- **Key contribution**: First compiler to automate the full pipeline from tensor programs to FHE execution

> Dathathri et al., "CHET: An Optimizing Compiler for Fully-Homomorphic Neural-Network Inferencing," PLDI 2019 [[3]](#ref3)

### 2.3 EVA (Microsoft Research, 2020)

A general-purpose compiler for CKKS programs with a Python frontend (PyEVA). EVA automatically inserts relinearization and rescaling operations, performs optimizations, and selects encryption parameters.

- **Scheme**: CKKS (via SEAL)
- **Optimizations**: Automatic waterline rescaling, lazy relinearization, common subexpression elimination, reduction balancing
- **Performance**: Programs compiled with EVA are on average **5.3x faster** than those from CHET
- **Key contribution**: Made CKKS programming accessible to non-cryptographers

> Dathathri et al., "EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation," PLDI 2020 [[4]](#ref4)

An improved version (EVA Improved) added a richer Python API and an Extension Library (EXL) with vector/matrix classes.

> Chielle et al., "EVA Improved: Compiler and Extension Library for CKKS," ePrint 2021/1505 [[5]](#ref5)

### 2.4 nGraph-HE (Intel, 2018)

An extension of Intel's nGraph deep learning compiler that treats HE as a hardware backend. It allows deploying TensorFlow models on encrypted data without requiring cryptographic expertise.

- **Scheme**: BFV/CKKS (via SEAL)
- **Optimizations**: HE-aware compile-time optimizations (constant folding, HE-SIMD packing), runtime optimizations (special-value plaintext bypass)
- **Performance**: Demonstrated MNIST and MobileNetV2/ImageNet inference
- **Key contribution**: First to integrate FHE into a mainstream ML compiler framework

> Boemer et al., "nGraph-HE: A Graph Compiler for Deep Learning on Homomorphically Encrypted Data," ePrint 2019/350 [[6]](#ref6)

### 2.5 Orion (NYU, 2023)

A fully-automated compiler that accepts PyTorch neural networks and produces optimized FHE programs. Orion represents the current state-of-the-art for FHE deep learning compilation.

- **Scheme**: CKKS with bootstrapping
- **Optimizations**: Single-shot multiplexed packing for arbitrary convolutions, automated bootstrap placement and scale management
- **Performance**:
  - **2.38x speedup** over prior state-of-the-art on ResNet-20
  - First to evaluate **ResNet-50 on ImageNet** under FHE
  - First **YOLO-v1 object detection** (139M parameters) under FHE
- **Key contribution**: Scaled FHE NN inference to production-sized models

> Ebel, Garimella, and Reagen, "Orion: A Fully Homomorphic Encryption Framework for Deep Learning," arXiv:2311.03470, ASPLOS 2025 [[7]](#ref7)

A follow-up extends Orion to **Transformer inference**, introducing BSGS-based linear algebra kernels (13.7x speedup over packed-row methods) and network-level pruning (up to 11.46x reduction in feed-forward layer runtimes).

> Ebel et al., "Extending Orion for Transformer Inference," arXiv:2512.11135 [[8]](#ref8)

### 2.6 Concrete / Concrete ML (Zama, 2021–present)

An FHE compiler ecosystem based on TFHE with programmable bootstrapping. Unlike CKKS-based compilers, Concrete uses lookup tables for non-linear functions, avoiding polynomial approximation entirely.

- **Scheme**: TFHE (programmable bootstrapping)
- **Optimizations**: Automatic quantization (QAT via Brevitas, post-training quantization), FHE-compatible model compilation from PyTorch/ONNX, automatic cryptographic parameter selection
- **Performance**: Demonstrated NN-20, NN-50, NN-100 architectures. July 2024 update beat their original 2021 paper benchmarks significantly.
- **Key contribution**: Alternative paradigm — TFHE's programmable bootstrapping enables exact non-linear functions instead of polynomial approximations

> Zama, "Making FHE Faster for ML: Beating our Previous Paper Benchmarks with Concrete ML," 2024 [[9]](#ref9)

### 2.7 HEIR (Google, 2024–present)

A universal FHE compiler built on the MLIR framework, designed as an open infrastructure for FHE research and development.

- **Scheme**: Supports all mainstream schemes (BFV, BGV, CKKS, TFHE) via OpenFHE and Lattigo backends
- **Optimizations**: Modular pass-based optimization infrastructure, supports hardware acceleration targets (GPU, TPU, FPGA, ASIC)
- **Key contribution**: Emerging standard infrastructure for FHE compilation research

> Goyal et al., "HEIR: A Universal Compiler for Homomorphic Encryption," arXiv:2508.11095 [[10]](#ref10)

### 2.8 Additional Compilers

| Compiler | Year | Key Focus | Reference |
|---|---|---|---|
| **Gazelle** | 2018 | Hybrid HE + garbled circuits for low-latency NN inference. 20x faster than MiniONN, 1000x faster than CryptoNets. | Juvekar et al., USENIX Security 2018 [[11]](#ref11) |
| **HECO** | 2023 | End-to-end compiler from imperative programs to FHE. Addresses full development pipeline, not just crypto optimizations. | Viand et al., USENIX Security 2023 [[12]](#ref12) |
| **Porcupine** | 2021 | Program synthesis-based compiler. Up to 51% speedup over hand-optimized HE kernels. | Cowan et al., arXiv:2101.07841 [[13]](#ref13) |
| **Fhelipe** | 2024 | Tensor compiler with automatic data packing. 18.5x faster than prior FHE compilers on average. | Malik et al., PLDI 2024 [[14]](#ref14) |
| **HECATE** | 2022 | Performance-aware scale optimization for CKKS. | Lee et al., IEEE CGO 2022 [[15]](#ref15) |
| **AutoFHE** | 2023 | Automated mixed-degree polynomial activations + bootstrapping placement via multi-objective optimization. 1.32–1.8x speedup. | Ao and Boddeti, USENIX Security 2024 [[16]](#ref16) |

---

## 3. Optimization Techniques

### 3.1 Ciphertext Packing Strategies

The way data is arranged inside ciphertext slots is the single most impactful optimization for FHE NN performance. Each CKKS/BFV ciphertext has N/2 slots (e.g., 4096 slots for N=8192), and SIMD operations apply to all slots in parallel.

| Strategy | Description | Best For |
|---|---|---|
| **Dense Packing** | Pack entire feature maps contiguously into slots | Fully-connected layers |
| **Convolution Packing** | Arrange input channels for convolution-friendly access patterns | Standard convolutions |
| **Multiplexed Packing** (Orion) | Pack multiple channels into a single ciphertext using a single-shot strategy | Arbitrary convolutions, deep CNNs |
| **Hybrid Packing** (HyPHEN) | Combine 2D gap packing with PRCR scheme to minimize rotations and bootstraps | ResNet-scale models |
| **Automatic Packing** (Fhelipe) | Compiler analytically selects optimal layout from a wide range of options | General tensor programs |

> Kim et al., "HyPHEN: A Hybrid Packing Method and Its Optimizations for Homomorphic Encryption-Based Neural Networks," IEEE Access 2024 [[17]](#ref17)
>
> Malik et al., "A Tensor Compiler with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption," PLDI 2024 [[14]](#ref14)

### 3.2 Convolution Optimization

Convolutions are the most expensive linear operation in FHE CNNs due to the massive number of ciphertext rotations required.

**Key techniques:**

- **Baby-Step Giant-Step (BSGS)**: Decomposes matrix-vector products into two nested loops of rotations, reducing rotation count from O(n) to O(√n). Achieves **13.7x speedup** over naive packed-row methods at transformer scale.

  > Ebel et al., arXiv:2512.11135 [[8]](#ref8)

- **Low-Rank Factorization (FFConv)**: Decomposes d×d convolutions into low-rank factorized convolutions followed by 1×1 convolutions. Reduces rotation overhead from both Dense Packing and Convolution Packing simultaneously. Achieves **up to 88% latency reduction** compared to prior methods.

  > Lee and Lee, "Optimizing HE-based CNN Inference via Low-Rank Factorized Convolutions," ePrint 2021 [[18]](#ref18)

- **Rotation Reduction (Rotom)**: Systematic autovectorization framework that explores layout assignments and applies rotation minimization, achieving **up to 4x reduction** in rotations.

  > "Bridging Usability and Performance: A Tensor Compiler for Autovectorizing Homomorphic Encryption," ePrint 2025/1319 [[19]](#ref19)

### 3.3 Activation Function Handling

Since FHE only supports additions and multiplications, non-linear activations must be handled specially:

| Approach | Description | Accuracy Impact | Speed Impact |
|---|---|---|---|
| **Square activation** (x²) | Replace ReLU with x². Minimal depth cost (1 multiplication). | Significant accuracy loss | Fastest |
| **Low-degree polynomial** (deg 2–4) | Approximate ReLU with a fitted polynomial. | Moderate accuracy loss | Fast |
| **High-degree polynomial** (deg 10–27) | More accurate approximation but requires bootstrapping between polynomial evaluations. | Minimal accuracy loss | Slow (bootstrapping overhead) |
| **Mixed-degree** (AutoFHE) | Different polynomial degrees per layer, jointly optimized with bootstrapping placement. | Best accuracy/speed tradeoff | Moderate |
| **Scheme switching** | Switch between CKKS (for linear ops) and TFHE/FHEW (for exact non-linear ops). | Near-exact | Very slow (switching cost) |
| **Programmable bootstrapping** (TFHE) | Evaluate non-linear functions as lookup tables during bootstrapping itself. | Exact | Moderate (bootstrapping is the operation) |
| **RBOOT fusion** | Fuse ReLU evaluation into CKKS bootstrapping by leveraging trigonometric functions in EvalMod. 2.77x faster, 81% less memory. | Good | Fast (no extra depth) |

> Ao and Boddeti, "AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE," ePrint 2023/162 [[16]](#ref16)
>
> Aharoni et al., "RBOOT: Accelerating Homomorphic Neural Network Inference by Fusing ReLU within Bootstrapping," ePrint 2025/1534 [[20]](#ref20)
>
> Bae et al., "Activate Me!: Designing Efficient Activation Functions for Privacy-Preserving Machine Learning with FHE," arXiv:2508.11575 [[21]](#ref21)

### 3.4 Bootstrapping Placement

Bootstrapping refreshes the noise budget, allowing deeper computations. Its placement is critical because it is extremely expensive (10³–10⁵x slower than a single multiplication).

- **Orion**: Automated bootstrap scheduling that minimizes the number of bootstraps needed while maintaining correctness
- **AutoFHE**: Multi-objective optimization jointly placing bootstraps and selecting activation polynomial degrees
- **HEAP**: Hybrid scheme-switching (CKKS ↔ TFHE) for parallelized bootstrapping, achieving 15.39x improvement over prior accelerators
- **FHE-Agent**: LLM-guided agentic framework that automates CKKS configuration including bootstrapping schedules

> Lu et al., "FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework," arXiv:2511.18653 [[22]](#ref22)

### 3.5 Quantization for FHE

Reducing the precision of NN weights and activations directly benefits FHE performance by lowering noise consumption and enabling smaller parameters.

| Technique | Bit Width | FHE Impact |
|---|---|---|
| Post-training quantization | 8–16 bits | Moderate speedup |
| Quantization-aware training (QAT) | 2–8 bits | Significant speedup, 80% execution time reduction reported for MNIST |
| Binary/ternary networks | 1–2 bits | Maximum speedup but significant accuracy loss |

> Legiest et al., "Optimizing FHE NN Inference through Quantization-Aware Training," arXiv:2304.09490 [[23]](#ref23)
>
> Zama Concrete ML uses Brevitas for QAT [[9]](#ref9)

### 3.6 Network Architecture Modifications

Several works modify the NN architecture itself to be FHE-friendly:

- **Replace max pooling with average pooling** (average pooling is a linear operation, free in FHE; max pooling requires comparisons which are extremely expensive)
- **Replace batch normalization** by folding BN parameters into preceding convolution weights (a standard technique that eliminates runtime non-linearity)
- **Reduce network depth** through knowledge distillation to minimize multiplicative depth
- **Structured pruning** to reduce the number of operations, especially in attention layers (up to 11.46x speedup in feed-forward layers)

> Ebel et al., arXiv:2512.11135 [[8]](#ref8); Ao and Boddeti, ePrint 2023/162 [[16]](#ref16)

---

## 4. Performance Benchmarks

### 4.1 Inference Latency Comparison

| System | Model | Dataset | Latency | Accuracy | Year |
|---|---|---|---|---|---|
| CryptoNets | Custom CNN | MNIST | ~0.06s (amortized) | 99.0% | 2016 |
| LoLa | Custom CNN | MNIST | 2.2s | 98.9% | 2019 |
| Gazelle | Custom CNN | MNIST | 0.03s | 99.0% | 2018 |
| nGraph-HE | MobileNetV2 | ImageNet | minutes | — | 2019 |
| CHET | SqueezeNet | ImageNet | minutes | — | 2019 |
| Cheetah (ASIC) | ResNet-50 | ImageNet | 0.03s | — | 2022 |
| HyPHEN | ResNet-20 | CIFAR-10 | **1.4s** | 92.4% | 2024 |
| HyPHEN | ResNet-18 | ImageNet | **14.7s** | — | 2024 |
| Orion | ResNet-20 | CIFAR-10 | ~seconds | 92%+ | 2023 |
| Orion | ResNet-50 | ImageNet | minutes | — | 2023 |
| Orion | YOLO-v1 | Detection | — | — | 2023 |
| FHEON | LeNet-5 | MNIST | 13s | 98.5% | 2024 |
| FHEON | ResNet-20 | CIFAR-10 | 403s | 92.2% | 2024 |
| Encrypted ResNet-20 | ResNet-20 | CIFAR-10 | ~5 min (CPU) | 91.7% | 2024 |

> Sources: [[2]](#ref2)[[7]](#ref7)[[11]](#ref11)[[17]](#ref17)[[24]](#ref24)[[25]](#ref25)

### 4.2 Underlying FHE Libraries

| Library | Maintainer | Schemes | Used By |
|---|---|---|---|
| **SEAL** | Microsoft | BFV, BGV, CKKS | CHET, EVA, nGraph-HE, SEALion |
| **OpenFHE** | Duality/DARPA | BFV, BGV, CKKS, FHEW, TFHE | HEIR, research community |
| **HElib** | IBM | BGV, CKKS | Academic research |
| **HEAAN** | CryptoLab | CKKS | CHET (original) |
| **Lattigo** | Tune Insight | BFV, BGV, CKKS, TFHE | HEIR |
| **TFHE-rs** | Zama | TFHE | Concrete, Concrete ML |
| **Concrete** | Zama | TFHE | Concrete ML, industry users |

---

## 5. Best-in-Class Methods per NN Operation (Practical Cookbook)

This section provides the **current best method for optimizing each neural network operation** under FHE, with links to papers and code. The idea is that these known-best transformations can be applied as a **preprocessing pass before the TRS/RL agent** — the agent should not have to rediscover well-established algebraic optimizations.

### 5.1 Convolution

Convolutions are the most expensive linear operation due to the massive number of ciphertext rotations required by packing schemes.

| Method | Key Idea | Speedup | Code | Reference |
|---|---|---|---|---|
| **Multiplexed Packing** (Orion) | Pack multiple channels into one ciphertext using a single-shot strategy for arbitrary convolutions. Eliminates per-channel overhead. | **2.38x** over SOTA on ResNet-20 | [github.com/baahl-nyu/orion](https://github.com/baahl-nyu/orion) | Ebel et al., ASPLOS 2025 [[7]](#ref7) |
| **HyPHEN RAConv/CAConv** | Alternate between two convolution algorithms (Row-Accumulation and Column-Accumulation) to minimize rotations. Hybrid gap packing for sparse-to-dense conversion. | **3.4–4.4x** over prior; 1.4s for ResNet-20 | Paper: IEEE Access 2024 | Kim et al., IEEE Access 2024 [[17]](#ref17) |
| **FFConv (Low-Rank Factorization)** | Decompose d×d convolution into low-rank d×d + 1×1 convolution. Reduces rotations from both DensePack and ConvPack simultaneously. | **88% latency reduction** vs LoLa | Paper: arXiv:2102.03494 | Lee and Lee, ePrint 2021 [[18]](#ref18) |
| **Depthwise-Separable Conv** (FastFHE) | Replace standard convolutions with depthwise-separable variants (depthwise + pointwise). Dramatically reduces operation count. | **2.41x** latency reduction on ResNet-20 | Paper: arXiv:2511.22434 | FastFHE, 2024 [[32]](#ref32) |

**Recommendation for Chehab:** Implement Orion's multiplexed packing as the default convolution strategy (open-source, PyTorch-native). For very deep networks, combine with FFConv's low-rank factorization as a preprocessing step that reduces convolution complexity before the RL agent optimizes the resulting expression tree.

### 5.2 Fully Connected / Linear Layers

Matrix-vector products are the core of FC layers and attention mechanisms.

| Method | Key Idea | Speedup | Code | Reference |
|---|---|---|---|---|
| **Diagonal Method** (Halevi-Shoup) | Pack matrix diagonals into ciphertexts; use rotations to align for multiplication. Multiplicative depth = 1. | Foundational method | [github.com/j2kun/fhe-packing](https://github.com/j2kun/fhe-packing) | Halevi & Shoup, ePrint 2018/244 [[33]](#ref33) |
| **BSGS (Baby-Step Giant-Step)** | Decompose diagonal method into two nested loops of sqrt(n) rotations instead of n. Reduces rotation count from O(n) to O(√n). | **13.7x** over packed-row at transformer scale | [github.com/baahl-nyu/orion](https://github.com/baahl-nyu/orion) | Ebel et al., arXiv:2512.11135 [[8]](#ref8) |
| **MAT-FHE** | Reduce KeySwitch operations from O(d) to O(log d) for arbitrary-dimension matrices using SIMD. | **1.5x** over diagonal method | Paper: Springer 2024 | Li et al., Cybersecurity 2024 [[34]](#ref34) |

**Recommendation for Chehab:** BSGS is the clear winner for FC/attention layers. It is already implemented in Orion and OpenFHE. Any matrix-vector product in the NN should be converted to BSGS form before going to the RL agent.

### 5.3 Activation Functions (ReLU, SiLU, etc.)

The most critical non-linear challenge. FHE only supports additions and multiplications natively.

| Method | Key Idea | Accuracy | Speed | Code | Reference |
|---|---|---|---|---|---|
| **SmartPAF** | Four techniques: Coefficient Tuning, Progressive Approximation, Alternate Training, Dynamic/Static Scaling. Recovers accuracy from low-degree polynomials. | **1.42–13.64x accuracy improvement** over prior PAFs | **6.79–14.9x** speedup over high-degree approx | [github.com/EfficientFHE/SmartPAF](https://github.com/EfficientFHE/SmartPAF) | Kim et al., MLSys 2024 [[35]](#ref35) |
| **AutoFHE (Mixed-Degree)** | Different polynomial degrees per layer, jointly optimized with bootstrapping placement via multi-objective search. | +2.56% over low-degree | **1.32–1.8x** speedup over high-degree | [github.com/human-analysis/autofhe](https://github.com/human-analysis/autofhe) | Ao & Boddeti, USENIX Security 2024 [[16]](#ref16) |
| **RBOOT (Fused ReLU)** | Evaluate ReLU inside CKKS bootstrapping by leveraging trigonometric functions in EvalMod. No extra depth needed. | Good | **2.77x** faster, 81% less memory | Paper: ePrint 2025/1534 | Aharoni et al., ePrint 2025 [[20]](#ref20) |
| **Low-Degree Legendre** (FastFHE) | Approximate SiLU with low-degree Legendre polynomials tuned for the specific input distribution. | Within ±1% of plaintext | Low depth cost | Paper: arXiv:2511.22434 | FastFHE, 2024 [[32]](#ref32) |
| **Scheme Switching** (CKKS↔TFHE) | Switch to TFHE for exact lookup-table evaluation of ReLU, then switch back to CKKS for linear ops. | Near-exact | Slow (switching overhead) | Paper: arXiv:2508.11575 | Bae et al., 2025 [[21]](#ref21) |
| **Programmable Bootstrapping** (TFHE) | Use TFHE's bootstrapping to apply arbitrary functions as lookup tables. Exact, no approximation needed. | Exact | Moderate | [docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml) | Zama Concrete ML [[9]](#ref9) |

**Recommendation for Chehab:** For a CKKS-based pipeline, use **SmartPAF** (open-source, best accuracy/speed tradeoff for low-degree polynomials) as a preprocessing step to replace activations, then let the RL agent optimize the resulting polynomial expressions. For a TFHE-based pipeline, Zama's programmable bootstrapping avoids the approximation problem entirely.

### 5.4 Batch Normalization

BN layers contain non-linear operations (division by variance) that are expensive in FHE.

| Method | Key Idea | Impact | Reference |
|---|---|---|---|
| **Weight Folding** (standard) | During model preparation (before encryption), absorb BN parameters (mean, variance, gamma, beta) into the preceding Conv/FC layer weights: `W_folded = gamma * W / sqrt(var + eps)`, `b_folded = gamma * (b - mean) / sqrt(var + eps) + beta`. This is done in plaintext — zero FHE cost. | **Eliminates BN entirely** from FHE circuit | Standard technique, used in all FHE NN compilers [[7]](#ref7)[[16]](#ref16)[[24]](#ref24) |
| **BN Dot-Product Fusion** (FastFHE) | Extended folding that merges BN with convolution as a fusion matrix, preserving the BN effect without extra multiplicative depth even in complex architectures. | **Zero extra depth** | FastFHE, arXiv:2511.22434 [[32]](#ref32) |

**Recommendation for Chehab:** Always fold BN into preceding layers as a mandatory preprocessing step. This is a pure algebraic transformation that should happen before any TRS/RL optimization — it's free and universally applied.

### 5.5 Pooling Layers

| Method | Key Idea | Impact | Reference |
|---|---|---|---|
| **Replace MaxPool → AveragePool** | MaxPool requires comparisons (extremely expensive in FHE — needs bootstrapping or bit decomposition). AveragePool is a simple weighted sum — it's a free linear operation. Retrain the model with AvgPool to recover accuracy. | **Eliminates non-linear pooling** | Standard in all FHE NN work [[1]](#ref1)[[24]](#ref24) |
| **Replace MaxPool → Strided Convolution** | Use a convolution with stride > 1 to achieve downsampling. More flexible than AvgPool and can be learned. | Eliminates pooling entirely; slightly higher accuracy | UniHENN, arXiv:2402.03060 [[36]](#ref36) |

**Recommendation for Chehab:** Replace all MaxPool layers with AveragePool or strided convolutions as a preprocessing step.

### 5.6 Bootstrapping Placement

Not a single operation, but a critical decision that must be inserted between layers when noise budget is exhausted.

| Method | Key Idea | Impact | Code | Reference |
|---|---|---|---|---|
| **Orion Automated Scheduler** | Jointly optimize bootstrap placement and scale management across the full network. | Enables ResNet-50 and YOLO-v1 under FHE | [github.com/baahl-nyu/orion](https://github.com/baahl-nyu/orion) | Ebel et al., ASPLOS 2025 [[7]](#ref7) |
| **AutoFHE Multi-Objective** | Treat bootstrap placement jointly with activation degree selection as a Pareto optimization problem. | 1.32–1.8x speedup over uniform-degree approach | [github.com/human-analysis/autofhe](https://github.com/human-analysis/autofhe) | Ao & Boddeti, USENIX Security 2024 [[16]](#ref16) |
| **FHE-Agent (LLM-Guided)** | Use an LLM agent to decompose the configuration search into global parameter selection + layer-wise bottleneck repair. | Finds configs where heuristics fail | Paper: arXiv:2511.18653 | Lu et al., 2025 [[22]](#ref22) |

**Recommendation for Chehab:** This is where the RL agent has the most value. Bootstrapping placement is a sequential decision problem with a noise budget constraint — exactly what Chehab's constrained RL agent is designed for. The preprocessing steps above (BN folding, activation replacement, convolution optimization) simplify the circuit, and then the RL agent decides optimal bootstrap insertion points.

### 5.7 Quantization (Pre-FHE Model Preparation)

Reducing precision before encryption reduces noise consumption and enables smaller/faster parameters.

| Method | Key Idea | Impact | Code | Reference |
|---|---|---|---|---|
| **Quantization-Aware Training** (Brevitas/Concrete ML) | Train the model with simulated quantization (2–8 bit weights/activations). Model learns to tolerate low precision. | **80% execution time reduction** for MNIST; 40% for CIFAR-10 | [github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas), [docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml) | Legiest et al., arXiv:2304.09490 [[23]](#ref23); Zama [[9]](#ref9) |
| **Post-Training Quantization** | Apply quantization after normal training. Simpler but less accurate at very low bit widths. | Moderate speedup | Concrete ML built-in | Zama [[9]](#ref9) |

**Recommendation for Chehab:** Apply QAT as the first step in the pipeline, before any FHE-specific transformations.

### 5.8 Summary: Recommended Preprocessing Pipeline for Chehab

The following pipeline applies known-best optimizations in order **before** the expression reaches the TRS/RL agent:

```
Step 1: Quantization-Aware Training (QAT)
   └─ Reduce precision to 4-8 bits using Brevitas
   └─ Lower noise consumption, smaller parameters

Step 2: Architecture Normalization
   └─ Fold BatchNorm into preceding Conv/FC weights (algebraic, free)
   └─ Replace MaxPool → AveragePool or Strided Convolution
   └─ Replace ReLU → Low-degree polynomial (SmartPAF or AutoFHE)

Step 3: Operation-Level Optimization
   └─ Convert convolutions → Multiplexed Packing (Orion) or FFConv
   └─ Convert FC layers → BSGS diagonal method
   └─ Apply network-level pruning to attention/FC layers

Step 4: Generate FHE Expression Tree
   └─ Compile optimized NN to FHE operations (CKKS or BFV)
   └─ Insert initial bootstrapping estimates

Step 5: TRS/RL Agent Optimization (Chehab)
   └─ Apply TRS rewrite rules to further reduce cost
   └─ RL agent optimizes bootstrap placement under noise constraint
   └─ RL agent explores algebraic simplifications the preprocessing missed
```

This two-phase approach lets established techniques handle the "known science" while the RL agent focuses on the harder combinatorial optimization that resists closed-form solutions.

---

## 6. Term Rewriting Approaches in FHE

This section is directly relevant to Chehab, which uses TRS (Term Rewriting Systems) to optimize FHE expressions.

### 5.1 Lobster: Program Synthesis + Term Rewriting (PLDI 2020)

The most closely related work to Chehab. Lobster automatically learns optimization rules for FHE circuits via program synthesis, then applies them through term rewriting to reduce multiplicative depth.

- **Approach**: Discover circuit equivalences from training data, generalize them into rewrite rules, apply via equational matching
- **Target**: Multiplicative depth reduction (the dominant performance factor)
- **Results**: 1.18x–3.71x speedup (geometric mean 2.05x), optimized 19 of 25 test circuits
- **Soundness**: Formally proven sound and terminating

> Lee et al., "Optimizing Homomorphic Evaluation Circuits by Program Synthesis and Term Rewriting," PLDI 2020 [[26]](#ref26)

A follow-up enhanced the approach with **equality saturation** — an exhaustive search over all possible rewrite orderings — achieving 1.08x–3.17x speedup (geometric mean 1.56x) on top of the original.

> Lee et al., "Optimizing Homomorphic Evaluation Circuits by Program Synthesis and Term Rewriting with Equality Saturation," TOPLAS 2023 [[27]](#ref27)

### 5.2 How This Relates to Chehab

Chehab applies TRS rules to optimize FHE expression trees with an RL agent choosing which rules to apply and where. The key differences from Lobster:

| Aspect | Lobster | Chehab |
|---|---|---|
| Rule discovery | Automatic (program synthesis) | Manual TRS rules |
| Rule application strategy | Equality saturation (exhaustive) | RL agent (learned policy) |
| Optimization target | Multiplicative depth only | Cost function (more general) |
| Constraint handling | None | Noise budget constraints |
| Scalability | Limited by synthesis/saturation | Scales with RL training |

---

## 7. Open Challenges

Based on the literature analysis, the following challenges remain open:

### 6.1 The Linear/Non-Linear Gap

No single FHE scheme handles both linear and non-linear operations efficiently. CKKS excels at linear algebra but requires polynomial approximations for activations. TFHE handles non-linear functions exactly via programmable bootstrapping but is slow on linear algebra. Scheme-switching between CKKS and TFHE is promising but incurs significant overhead.

> Lee et al., arXiv:2504.11604 [[1]](#ref1)

### 6.2 Bootstrapping Remains the Bottleneck

Bootstrapping is 10³–10⁵x slower than a single multiplication. Every activation function in a deep network may require a bootstrap, making deep networks extremely slow. Recent work (RBOOT, HEAP) reduces this overhead but it remains the dominant cost.

> Aharoni et al., ePrint 2025/1534 [[20]](#ref20)

### 6.3 No Standardized Building Blocks

The field lacks reusable, standardized FHE components for NN layers. Each compiler reimplements convolutions, activations, and pooling from scratch with different design choices.

> Viand et al., ePrint 2024/612 [[28]](#ref28)

### 6.4 Packing Is Still Largely Manual or Heuristic

While Fhelipe and Rotom automate packing decisions, most compilers still rely on hand-tuned or heuristic packing strategies that may not be optimal.

### 6.5 Limited Support for Training on Encrypted Data

Most work focuses on inference. Encrypted training (backpropagation under FHE) is orders of magnitude harder and remains impractical for large models.

> Montero et al., arXiv:2401.16136 [[29]](#ref29)

---

## 8. Implications for Chehab

Based on this survey, here are concrete directions for extending Chehab to support neural network workloads:

### 7.1 Expand the TRS Rule Set for NN Operations

Current Chehab rules target general FHE expression trees. To support NNs, add rewrite rules for:

- **Convolution decomposition**: Rules to convert expensive convolution patterns into rotation-efficient equivalents (e.g., factorizing d×d convolutions into sequences of 1×d and d×1 operations, as in FFConv)
- **Activation replacement**: Rules to substitute high-degree polynomial approximations with lower-degree alternatives when noise budget allows
- **Batch norm folding**: Rules to absorb BN parameters into preceding convolution/linear layer weights
- **Pooling conversion**: Rules to replace max-pool patterns with average-pool equivalents

### 7.2 Add Packing-Aware Optimization

The RL agent currently optimizes expression trees without considering how data is packed into ciphertexts. For NN workloads, packing decisions dramatically affect performance:

- Extend the expression representation to include packing/layout annotations
- Add TRS rules that transform between packing strategies (dense ↔ convolution ↔ multiplexed)
- Train the RL agent to jointly optimize algebraic rewrites and packing decisions

### 7.3 Bootstrapping Placement as a Constraint

Similar to the noise budget constraint, bootstrapping placement can be modeled as an RL decision:

- The agent decides when to insert bootstrapping operations
- Each bootstrap has a fixed (high) cost but resets the noise budget
- The optimization becomes: minimize total cost (computation + bootstrapping) subject to noise budget constraints at every point in the circuit

This directly extends Chehab's existing constrained RL framework.

### 7.4 Target CKKS (Not Just BFV)

Most NN-focused FHE compilers use CKKS for its approximate arithmetic, which naturally matches floating-point NN computations. Extending Chehab's noise estimator and TRS rules to the CKKS scheme would align with the dominant paradigm for FHE ML.

### 7.5 Layer-Level Optimization

Instead of optimizing individual expressions, operate at the **layer level**:

- Represent an entire convolutional layer as an expression tree
- Apply TRS rules that exploit cross-operation optimizations (e.g., fusing rotation sequences across adjacent multiplications)
- This aligns with how Orion and CHET achieve their best results — optimizing at the tensor program level rather than individual operations

### 7.6 Leverage RL for Compilation Decisions

Several works (FHE-Agent, AutoFHE) have shown that automated search over FHE compilation decisions is effective. Chehab's RL-based approach could be extended to make:

- **Packing layout decisions** per tensor
- **Polynomial degree selection** per activation layer
- **Bootstrapping insertion** decisions
- **Rotation sequence optimization** decisions

This would position Chehab as a **learned FHE compiler** — using RL where others use heuristics or brute-force search.

---

## 9. References

<a id="ref1"></a>[1] Lee et al., "SoK: Can Fully Homomorphic Encryption Support General AI Computation? A Functional and Cost Analysis," arXiv:2504.11604, 2025. https://arxiv.org/abs/2504.11604

<a id="ref2"></a>[2] Gilad-Bachrach et al., "CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy," ICML 2016. https://proceedings.mlr.press/v48/gilad-bachrach16.html

<a id="ref3"></a>[3] Dathathri et al., "CHET: An Optimizing Compiler for Fully-Homomorphic Neural-Network Inferencing," PLDI 2019. https://www.cs.utexas.edu/~roshan/CHET.pdf

<a id="ref4"></a>[4] Dathathri et al., "EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation," PLDI 2020. https://dl.acm.org/doi/10.1145/3385412.3386023

<a id="ref5"></a>[5] Chielle et al., "EVA Improved: Compiler and Extension Library for CKKS," ePrint 2021/1505. https://eprint.iacr.org/2021/1505

<a id="ref6"></a>[6] Boemer et al., "nGraph-HE: A Graph Compiler for Deep Learning on Homomorphically Encrypted Data," ePrint 2019/350. https://eprint.iacr.org/2019/350

<a id="ref7"></a>[7] Ebel, Garimella, and Reagen, "Orion: A Fully Homomorphic Encryption Framework for Deep Learning," arXiv:2311.03470, ASPLOS 2025. https://arxiv.org/abs/2311.03470

<a id="ref8"></a>[8] Ebel et al., "Extending Orion for Transformer Inference," arXiv:2512.11135, 2024. https://arxiv.org/abs/2512.11135

<a id="ref9"></a>[9] Zama, "Making FHE Faster for ML: Beating our Previous Paper Benchmarks with Concrete ML," 2024. https://zama.ai/post/making-fhe-faster-for-ml-beating-our-previous-paper-benchmarks-with-concrete-ml

<a id="ref10"></a>[10] Goyal et al., "HEIR: A Universal Compiler for Homomorphic Encryption," arXiv:2508.11095, 2025. https://arxiv.org/abs/2508.11095

<a id="ref11"></a>[11] Juvekar, Vaikuntanathan, and Chandrakasan, "GAZELLE: A Low Latency Framework for Secure Neural Network Inference," USENIX Security 2018. https://arxiv.org/abs/1801.05507

<a id="ref12"></a>[12] Viand et al., "HECO: Automatic Code Optimizations for Efficient Fully Homomorphic Encryption," USENIX Security 2023. https://www.usenix.org/system/files/usenixsecurity23-viand.pdf

<a id="ref13"></a>[13] Cowan et al., "Porcupine: A Synthesizing Compiler for Vectorized Homomorphic Encryption," arXiv:2101.07841, 2021. https://arxiv.org/abs/2101.07841

<a id="ref14"></a>[14] Malik et al., "A Tensor Compiler with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption," PLDI 2024. https://dl.acm.org/doi/10.1145/3656382

<a id="ref15"></a>[15] Lee et al., "HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler," IEEE CGO 2022. https://ieeexplore.ieee.org/document/9741265

<a id="ref16"></a>[16] Ao and Boddeti, "AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE," USENIX Security 2024. https://eprint.iacr.org/2023/162

<a id="ref17"></a>[17] Kim et al., "HyPHEN: A Hybrid Packing Method and Its Optimizations for Homomorphic Encryption-Based Neural Networks," IEEE Access 2024. https://ieeexplore.ieee.org/document/10376063

<a id="ref18"></a>[18] Lee and Lee, "Optimizing Packed HE-based CNN Inference via Low-Rank Factorized Convolutions," ePrint 2021. https://export.arxiv.org/abs/2102.03494

<a id="ref19"></a>[19] "Bridging Usability and Performance: A Tensor Compiler for Autovectorizing Homomorphic Encryption," ePrint 2025/1319. https://eprint.iacr.org/2025/1319

<a id="ref20"></a>[20] Aharoni et al., "RBOOT: Accelerating Homomorphic Neural Network Inference by Fusing ReLU within Bootstrapping," ePrint 2025/1534. https://eprint.iacr.org/2025/1534

<a id="ref21"></a>[21] Bae et al., "Activate Me!: Designing Efficient Activation Functions for Privacy-Preserving Machine Learning with FHE," arXiv:2508.11575, 2025. https://arxiv.org/abs/2508.11575

<a id="ref22"></a>[22] Lu et al., "FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework," arXiv:2511.18653, 2025. https://arxiv.org/abs/2511.18653

<a id="ref23"></a>[23] Legiest et al., "Accelerating Encrypted Neural Network Inference through Quantization-Aware Training," arXiv:2304.09490, 2023. https://arxiv.org/abs/2304.09490

<a id="ref24"></a>[24] "FHEON: A Configurable Framework for Developing Privacy-Preserving Neural Networks Using Homomorphic Encryption," arXiv:2510.03996, 2025. https://arxiv.org/abs/2510.03996

<a id="ref25"></a>[25] "Encrypted Image Classification with Low Memory Footprint using Fully Homomorphic Encryption," ePrint 2024/460. https://eprint.iacr.org/2024/460

<a id="ref26"></a>[26] Lee et al., "Optimizing Homomorphic Evaluation Circuits by Program Synthesis and Term Rewriting," PLDI 2020. https://dl.acm.org/doi/10.1145/3385412.3385996

<a id="ref27"></a>[27] Lee et al., "Optimizing Homomorphic Evaluation Circuits by Program Synthesis and Term Rewriting (with Equality Saturation)," TOPLAS 2023. https://psl.hanyang.ac.kr/assets/pdf/toplas23.pdf

<a id="ref28"></a>[28] Viand et al., "SoK: Fully Homomorphic Encryption Compilers," ePrint 2024/612. https://eprint.iacr.org/2024/612

<a id="ref29"></a>[29] Montero et al., "Neural Network Training on Encrypted Data with TFHE," arXiv:2401.16136, 2024. https://arxiv.org/abs/2401.16136

<a id="ref30"></a>[30] Shivdikar et al., "Cheetah: Optimizing and Accelerating Homomorphic Encryption for Private Inference," IEEE HPCA 2023. https://ieeexplore.ieee.org/document/10071018

<a id="ref31"></a>[31] "ARION: Attention-Optimized Transformer Inference on Encrypted Data," ePrint 2025/2271. https://eprint.iacr.org/2025/2271

<a id="ref32"></a>[32] "FastFHE: Packing-Scalable and Depthwise-Separable CNN Inference Over FHE," arXiv:2511.22434, 2024. https://arxiv.org/abs/2511.22434

<a id="ref33"></a>[33] Halevi and Shoup, "Faster Homomorphic Linear Transformations in HElib," ePrint 2018/244. https://eprint.iacr.org/2018/244

<a id="ref34"></a>[34] Li et al., "MAT-FHE: Arbitrary Dimension Matrix Multiplication Scheme for Floating Point over Fully Homomorphic Encryption," Cybersecurity, Springer 2024. https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00303-y

<a id="ref35"></a>[35] Kim et al., "SmartPAF: Accurate Low-Degree Polynomial Approximation of Non-Polynomial Operators for Fast Private Inference in Homomorphic Encryption," MLSys 2024. https://github.com/EfficientFHE/SmartPAF

<a id="ref36"></a>[36] "UniHENN: Designing Faster and More Versatile Homomorphic Encryption-based CNNs without im2col," arXiv:2402.03060, 2024. https://arxiv.org/abs/2402.03060

<a id="ref37"></a>[37] Kun, "Packing Matrix-Vector Multiplication in Fully Homomorphic Encryption," 2024. https://github.com/j2kun/fhe-packing
