# Microsoft SEAL BFV Noise Budget: Comprehensive Research

## 1. What Is the Noise Budget?

In the BFV (Brakerski/Fan-Vercauteren) scheme implemented in Microsoft SEAL, every ciphertext carries an internal quantity called the **invariant noise budget**, measured in **bits**. This budget represents how much "room" remains for homomorphic operations before the ciphertext becomes too corrupted to decrypt correctly.

> *"Each ciphertext has a specific quantity called the 'invariant noise budget' — or 'noise budget' for short — measured in bits. The noise budget in a freshly encrypted ciphertext (initial noise budget) is determined by the encryption parameters. Homomorphic operations consume the noise budget at a rate also determined by the encryption parameters."*
>
> — [SEAL BFV Basics Example (`1_bfv_basics.cpp`)](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp)

---

## 2. What Happens When the Noise Budget Reaches Zero?

### Silent Failure — No Error, No Exception

**When the noise budget is exhausted, SEAL does NOT throw an error.** The decryption operation still runs and produces output, but the decrypted plaintext is **silently incorrect** (corrupted garbage). There is no built-in warning mechanism.

> *"Decryption does not fail when the noise budget reaches zero—instead, it silently produces incorrect results. The decryption operation always succeeds and returns some output Plaintext, but when noise is exhausted, the decrypted values will be corrupted."*
>
> — [StackOverflow: Decoding and Decryption Error Indicators in SEAL](https://stackoverflow.com/questions/54962785/decoding-and-decryption-error-indicators-in-seal)

> *"SEAL does not provide automatic warnings or exceptions when the noise budget reaches zero. Users must manually check the noise budget separately."*
>
> — [StackOverflow: Decoding and Decryption Error Indicators in SEAL](https://stackoverflow.com/questions/54962785/decoding-and-decryption-error-indicators-in-seal)

The SEAL example code explicitly warns:

> *"NOTE: Decryption can be incorrect if noise budget is zero."*
>
> — [`1_bfv_basics.cpp`](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp)

### No Bootstrapping in BFV (Leveled Scheme)

BFV as implemented in SEAL is a **leveled** homomorphic encryption scheme. Once noise is exhausted, there is **no way to "refresh" or recover** the ciphertext without the secret key. This is fundamentally different from fully homomorphic schemes (like CKKS with bootstrapping) that can reset the noise.

> *"BFV is fundamentally designed as a leveled homomorphic encryption scheme with finite computation depth. The scheme operates under noise-based encryption, where homomorphic operations cause noise to grow in ciphertexts."*
>
> — [FHE Textbook: BFV Scheme](https://fhetextbook.github.io/BFVScheme.html)

While recent research has explored BFV bootstrapping ([Relaxed Functional Bootstrapping, ePrint 2024/172](https://eprint.iacr.org/2024/172)), it is **not implemented in SEAL** and remains impractical for most applications.

**Bottom line:** In BFV/SEAL, if noise budget reaches 0, the computation is irreversibly corrupted. The program does not crash — it just gives wrong answers.

---

## 3. How Is the Initial Noise Budget Determined?

### The Formula

The noise budget of a freshly encrypted ciphertext is approximately:

```
Initial Noise Budget ≈ log₂(coeff_modulus / plain_modulus)  bits
```

More precisely, since SEAL version 3.3:

```
Max Noise Budget = (Total coeff_modulus bits) − (Last prime bits) − (Plain modulus bits)
```

Where:
- **Total coeff_modulus bits** = sum of bit-lengths of all prime factors in the coefficient modulus
- **Last prime bits** = bit-length of the last prime in `coeff_modulus` (reserved for key-switching noise reduction since SEAL 3.3)
- **Plain modulus bits** = `log₂(plain_modulus)`

Fresh encryption consumes an additional ~0–10 bits.

> *"The noise budget in a freshly encrypted ciphertext is ~ log₂(coeff_modulus/plain_modulus) (bits) and the noise budget consumption in a homomorphic multiplication is of the form log₂(plain_modulus) + (other terms)."*
>
> — [SEAL BFV Basics Example](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp); also [Pyfhel BFV Demo](https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_2_Integer_BFV.html)

> *"Since version 3.3, the last element in the coefficient modulus is dedicated to noise reduction during key switching and is excluded from the noise budget calculation."*
>
> — [GitHub Issue #226: Questions about noise budget](https://github.com/microsoft/SEAL/issues/226)

---

## 4. SEAL Default Parameters and Resulting Noise Budgets

### Maximum coeff_modulus Bit-Length (128-bit Security)

The polynomial modulus degree constrains the maximum total coefficient modulus size while maintaining 128-bit security:

| `poly_modulus_degree` | Max `coeff_modulus` bit-length | Modulus count | Prime sizes |
|---|---|---|---|
| 1024 | 27 | 1 | 27 |
| 2048 | 54 | 1 | 54 |
| 4096 | 109 | 3 | 36 + 36 + 37 |
| 8192 | 218 | 5 | 43 + 43 + 44 + 44 + 44 |
| 16384 | 438 | 9 | 48×3 + 49×6 |
| 32768 | 881 | 16 | 55×15 + 56 |

> *Source: SEAL source code [`globals.cpp`](https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/globals.cpp) — `GetDefaultCoeffModulus128()` function; also documented in the [`1_bfv_basics.cpp` example](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp).*

These values can also be obtained at runtime via `CoeffModulus::MaxBitCount(poly_modulus_degree)`.

### Estimated Initial Noise Budgets

Using the formula `Max Budget = Total bits − Last prime bits − plain_modulus bits`:

| `poly_modulus_degree` | Total bits | Last prime | `plain_modulus` (t) | t bits | **Max Noise Budget** | **Fresh Budget (approx)** |
|---|---|---|---|---|---|---|
| 4096 | 109 | 37 | 1024 | 10 | 62 bits | ~55 bits |
| 4096 | 109 | 37 | 65537 | 17 | 55 bits | ~48 bits |
| 8192 | 218 | 44 | 1024 | 10 | 164 bits | ~155 bits |
| 8192 | 218 | 44 | 65537 | 17 | 157 bits | ~148 bits |
| 8192 | 218 | 44 | ~2⁴⁰ | 40 | 134 bits | ~127 bits |
| 16384 | 438 | 49 | 65537 | 17 | 372 bits | ~363 bits |
| 32768 | 881 | 56 | 65537 | 17 | 808 bits | ~800 bits |

> *Concrete example confirmed by SEAL developers: "With a poly_modulus_degree of 8192 and default BFV parameters: Coefficient modulus 218 bits, last prime 44 bits, plain modulus 40 bits → maximum budget 218 − 44 − 40 = 134 bits."*
>
> — [GitHub Issue #226](https://github.com/microsoft/SEAL/issues/226)

### Concrete Example from Pyfhel (SEAL Wrapper)

Using `n=8192`, `t_bits=20` (t=1032193), `sec=128`:

| Operation | Noise Budget (bits) |
|---|---|
| Fresh ciphertext | **146** |
| After addition (ctxt + ctxt) | **146** (essentially free) |
| After multiplication (ctxt × ctxt) | **114** (~32 bits consumed) |
| After ctxt³ (2 multiplications + relin) | **82** (~64 bits consumed) |
| After rotation | **143** (~3 bits consumed) |
| After plaintext multiply | **121** (~25 bits consumed) |

> *Source: [Pyfhel BFV Integer Demo](https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_2_Integer_BFV.html) — actual measured output.*

---

## 5. Noise Budget Consumption per Operation

### Summary Table

| Operation | Approx. Budget Consumed | Notes |
|---|---|---|
| **Addition** (ctxt + ctxt) | ~1 bit | Nearly free |
| **Addition** (ctxt + ptxt) | ~0 bits | Free |
| **Negation** (−ctxt) | 0 bits | Free |
| **Multiplication** (ctxt × ctxt) | ~30–35 bits | Dominant cost. Grows with `log₂(t)` |
| **Multiplication** (ctxt × ptxt) | ~25 bits | Cheaper than ctxt×ctxt |
| **Squaring** (ctxt²) | ~30–35 bits | Same as multiplication |
| **Relinearization** | ~0 bits (SEAL ≥ 3.3) | Nearly free since SEAL 3.3 |
| **Rotation** | ~3 bits | Nearly free since SEAL 3.3 |

> *"Additions can generally be thought of as being nearly free in terms of noise budget consumption compared to multiplications. Since noise budget consumption compounds in sequential multiplications, the most significant factor in choosing appropriate encryption parameters is the multiplicative depth."*
>
> — [`1_bfv_basics.cpp`](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp)

> *"Each homomorphic addition shrinks the noise budget by approximately 1 bit. Homomorphic multiplication consumes significantly more noise budget."*
>
> — [A Guide to Homomorphic Encryption Library SEAL](https://magic3007.github.io/blog/2020/05/12/A-Guide-to-Homomorphic-Encryption-Library-SEAL.html)

### The Multiplication Cost Formula

Each ciphertext-ciphertext multiplication consumes roughly:

```
Noise consumed ≈ log₂(plain_modulus) + log₂(poly_modulus_degree) + constant
```

This means:
- Larger `plain_modulus` → more noise consumed per multiplication → fewer multiplications possible
- The consumption **compounds** across sequential multiplications

---

## 6. Typical Multiplicative Depths in Practice

The multiplicative depth (number of sequential multiplications before noise is exhausted) depends on the parameter set:

| `poly_modulus_degree` | Approx. Multiplicative Depth | Use Case |
|---|---|---|
| 4096 | **~2** | Simple arithmetic, few multiplications |
| 8192 | **~4–6** | Moderate computations, polynomial evaluation |
| 16384 | **~10–15** | Complex circuits, machine learning inference |
| 32768 | **~20–30** | Deep circuits |

> *"poly_modulus_degree 4096: supports multiplicative depth of 2. poly_modulus_degree 8192: supports multiplicative depth of 5."*
>
> — [SEAL Manual v2.3.1](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/sealmanual-2-3-1.pdf) (depths are approximate and depend on plain_modulus)

The [SEAL-Depth-Estimator tool](https://github.com/WeiDaiWD/SEAL-Depth-Estimator) can estimate supported circuit depth for specific parameter choices.

---

## 7. How to Estimate Noise Budget from BFV Parameters

### Step-by-Step Estimation

Given the three core BFV parameters, you can estimate the available noise budget:

**Step 1:** Choose `poly_modulus_degree` (n) based on security needs.

**Step 2:** Determine `coeff_modulus` (Q). Either use `CoeffModulus::BFVDefault(n)` or select custom primes whose total bit-length does not exceed the maximum for your `n` (see table in Section 4).

**Step 3:** Choose `plain_modulus` (t) as small as possible while fitting your data range.

**Step 4:** Calculate:

```
Total Q bits  = sum of bit-lengths of all primes in coeff_modulus
Last prime    = bit-length of the last prime (reserved for key-switching)
t bits        = ceil(log₂(plain_modulus))

Max budget    = Total Q bits − Last prime bits − t bits
Fresh budget  ≈ Max budget − (5 to 10)    [fresh encryption noise]
```

**Step 5:** Estimate depth:

```
Budget per multiplication ≈ t bits + ceil(log₂(n)) + small constant (~5)
Multiplicative depth      ≈ floor(Fresh budget / Budget per multiplication)
```

### Worked Example

For a common configuration:
- `poly_modulus_degree` = 8192
- `coeff_modulus` = BFVDefault(8192) → 218 bits total, last prime = 44 bits
- `plain_modulus` = 65537 (17 bits)

```
Max budget   = 218 − 44 − 17 = 157 bits
Fresh budget ≈ 157 − 7 = 150 bits
Cost per mul ≈ 17 + 13 + 5 = ~35 bits
Depth        ≈ 150 / 35 ≈ 4 multiplications
```

This aligns with the documented ~4–5 multiplicative depth for `n=8192`.

---

## 8. Typical Noise Budget Ranges in Practice

Based on the parameter tables and real-world usage:

| Scenario | Fresh Noise Budget Range |
|---|---|
| **Minimal** (n=4096, large t) | 30–55 bits |
| **Standard** (n=8192, moderate t) | 120–165 bits |
| **Large** (n=16384, small t) | 350–380 bits |
| **Very large** (n=32768, small t) | 750–810 bits |

For **most practical BFV applications** with 128-bit security, the fresh noise budget falls in the range of **50–200 bits**, with `n=8192` being the most commonly used parameter set.

> *Confirmed by Pyfhel output: fresh ciphertext with n=8192 and t_bits=20 → **146 bits** noise budget.*
>
> — [Pyfhel BFV Demo](https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_2_Integer_BFV.html)

---

## 9. Relevance to Our RL Agent's Noise Budget Constraint

### Our NoiseEstimator Is Calibrated in SEAL Bits

The `NoiseEstimator` in our codebase (`pytrs/noise_estimator.py`) is a linear regression model trained on **real SEAL measurements**. The training dataset (`noise_estimator_stats_dataset.csv`) contains expressions with their actual `Remaining_noise_budget` as measured by SEAL's `invariant_noise_budget()`. The initial total budget in that dataset is **369 bits**.

The estimator predicts `used_noise = 369 − Remaining_noise_budget`, meaning **our noise values are in SEAL noise-budget bits**.

### The Dominant Factor: Multiplicative Depth

The learned model coefficients reveal that **Multiplicative Depth** is overwhelmingly the dominant noise driver (~33.4 bits per multiplicative depth level). All other operations (add, sub, rotate, multiply count) have near-zero coefficients:

| Feature              | Coefficient |
|---|---|
| Multiplicative Depth | **33.383** |
| Depth                | 1.198 |
| negate               | 1.028 |
| add, sub, multiply, rotate, multiply_plain | ~0.0 |

### What Noise Values Look Like Per Expression

| Expression Type | Estimated Noise (bits consumed) |
|---|---|
| Additions only (depth 1–2) | ~1–3 |
| 1 multiplication (multdepth 1) | ~35 |
| Multdepth 2 | ~70 |
| Multdepth 3 | ~105 |
| Multdepth 4 | ~140 |
| Multdepth 5+ | ~175+ |

### Distribution in Training Data

| Percentile | Noise Consumed (bits) |
|---|---|
| P10 | 41 |
| P25 | 42 |
| **P50 (median)** | **73** |
| P75 | 74 |
| P90 | 105 |
| P95 | 106 |
| P99 | 140 |
| Max | 369 |

Most expressions cluster around multdepth 1 (~41 bits) and multdepth 2 (~73 bits).

### What Our Budget Thresholds Mean

Given that the total SEAL budget is 369 bits and most expressions use 40–105 bits:

| Budget Threshold | Meaning |
|---|---|
| **60–80** | Extremely tight — only multdepth-1 expressions (additions + 1 multiplication) fit. Multdepth-2 expressions are forbidden. |
| **100–140** | Tight — multdepth-2 expressions fit, multdepth-3 are borderline. |
| **180–240** | **Moderate / realistic** — allows multdepth 3–5. Constrains only the top few percent of complex expressions. This is the practical "sweet spot" for training. |
| **300** | Permissive — essentially unconstrained for 99%+ of real expressions. |
| **369** | The absolute maximum (full SEAL budget). |
| **9,000,000** | Placeholder for "no constraint at all". |

### Key Takeaway

The noise budget is **not a fixed property of the encryption scheme alone** — it is the *gap* between the initial budget (set by SEAL parameters, here 369 bits) and the noise *consumed by the specific expression* (determined by its operations, especially multiplicative depth). Our RL agent's job is to rewrite expressions so their consumed noise stays below the given threshold, because in real SEAL, exceeding it means **silent decryption failure**.

---

## 10. Exact Training Budget Derivation

This section derives the **5 exact noise budget thresholds** used for constrained RL agent training. Each constrained budget corresponds to a real SEAL BFV parameter set taken directly from the official SEAL source code. The coefficient modulus primes come from `globals.cpp`, and the `plain_modulus` values are standard choices used in BFV applications.

### Source

All coefficient modulus primes are read verbatim from:

> [`native/src/seal/util/globals.cpp`](https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/globals.cpp)
>
> Functions: `GetDefaultCoeffModulus128()`, `GetDefaultCoeffModulus192()`, `GetDefaultCoeffModulus256()`

The formula used is (see Section 3):

```
Max Noise Budget = (Total coeff_modulus bits) − (Last prime bits) − ceil(log₂(plain_modulus))
```

The "last prime subtracted" rule (SEAL ≥ 3.3) is confirmed by [SEAL Issue #226](https://github.com/microsoft/SEAL/issues/226).

### Budget 1: 172 bits — `n=16384, 256-bit security, t=65537`

From `GetDefaultCoeffModulus256()`:

```cpp
{ 16384, { 0x7ffffffc8001, 0x7ffffff00001, 0x7fffffe70001, 0xfffffffd8001, 0xfffffffa0001 } }
//         Total bit count: 237 = 3 * 47 + 2 * 48
```

| Prime (hex) | Bit-length | Role |
|---|---|---|
| `0x7ffffffc8001` | 47 | Data level |
| `0x7ffffff00001` | 47 | Data level |
| `0x7fffffe70001` | 47 | Data level |
| `0xfffffffd8001` | 48 | Data level |
| `0xfffffffa0001` | 48 | Key-switching (last) |

**Calculation:**

```
Total Q       = 3×47 + 2×48 = 237 bits
Last prime    =                48 bits
t = 65537     → ceil(log₂(t)) = 17 bits
Max Budget    = 237 − 48 − 17 = 172 bits
```

MultDepth threshold: 172 / 33.38 ≈ **5.2** — constrains expressions with multiplicative depth ≥ 6.

`t = 65537` is the most standard BFV `plain_modulus` (Fermat prime F₄). It is batching-compatible with `n=16384` since `65537 ≡ 1 (mod 32768)`.

### Budget 2: 230 bits — `n=16384, 192-bit security, t=786433`

From `GetDefaultCoeffModulus192()`:

```cpp
{ 16384, { 0x3ffffffdf0001, 0x3ffffffd48001, 0x3ffffffd20001,
           0x3ffffffd18001, 0x3ffffffcd0001, 0x3ffffffc70001 } }
//         Total bit count: 300 = 6 * 50
```

| Prime (hex) | Bit-length | Role |
|---|---|---|
| `0x3ffffffdf0001` | 50 | Data level |
| `0x3ffffffd48001` | 50 | Data level |
| `0x3ffffffd20001` | 50 | Data level |
| `0x3ffffffd18001` | 50 | Data level |
| `0x3ffffffcd0001` | 50 | Data level |
| `0x3ffffffc70001` | 50 | Key-switching (last) |

**Calculation:**

```
Total Q       = 6 × 50      = 300 bits
Last prime    =                50 bits
t = 786433    → ceil(log₂(t)) = 20 bits
Max Budget    = 300 − 50 − 20 = 230 bits
```

MultDepth threshold: 230 / 33.38 ≈ **6.9** — constrains expressions with multiplicative depth ≥ 7.

`t = 786433` is a 20-bit batching prime: `786433 ≡ 1 (mod 32768)`, so it supports SIMD batching with `n=16384`. It is the standard output of `PlainModulus::Batching(16384, 20)`. This larger plaintext modulus supports computations on integers up to ±393,216.

### Budget 3: 236 bits — `n=16384, 192-bit security, t=12289`

Same coefficient modulus as Budget 2 (`GetDefaultCoeffModulus192()` for `n=16384`):

```cpp
{ 16384, { 0x3ffffffdf0001, 0x3ffffffd48001, 0x3ffffffd20001,
           0x3ffffffd18001, 0x3ffffffcd0001, 0x3ffffffc70001 } }
//         Total bit count: 300 = 6 * 50
```

**Calculation:**

```
Total Q       = 6 × 50      = 300 bits
Last prime    =                50 bits
t = 12289     → ceil(log₂(t)) = 14 bits
Max Budget    = 300 − 50 − 14 = 236 bits
```

MultDepth threshold: 236 / 33.38 ≈ **7.1** — constrains expressions with multiplicative depth ≥ 8.

`t = 12289` is a well-known NTT-friendly prime widely used in lattice-based cryptography (NTRU, NewHope). It is 14 bits. Note: `12289 mod 32768 = 12289 ≠ 1`, so this configuration does **not** support SIMD batching with `n=16384`. This represents a non-batching BFV deployment where each ciphertext encodes a single polynomial over `Z₁₂₂₈₉[x]`, which is valid and used in applications where batching is not required.

### Budget 4: 369 bits — `n=16384, 128-bit security, t=786433`

From `GetDefaultCoeffModulus128()`:

```cpp
{ 16384, { 0xfffffffd8001, 0xfffffffa0001, 0xfffffff00001,
           0x1fffffff68001, 0x1fffffff50001, 0x1ffffffee8001,
           0x1ffffffea0001, 0x1ffffffe88001, 0x1ffffffe48001 } }
//         Total bit count: 438 = 3 * 48 + 6 * 49
```

| Prime (hex) | Bit-length | Role |
|---|---|---|
| `0xfffffffd8001` | 48 | Data level |
| `0xfffffffa0001` | 48 | Data level |
| `0xfffffff00001` | 48 | Data level |
| `0x1fffffff68001` | 49 | Data level |
| `0x1fffffff50001` | 49 | Data level |
| `0x1ffffffee8001` | 49 | Data level |
| `0x1ffffffea0001` | 49 | Data level |
| `0x1ffffffe88001` | 49 | Data level |
| `0x1ffffffe48001` | 49 | Key-switching (last) |

**Calculation:**

```
Total Q       = 3×48 + 6×49 = 438 bits
Last prime    =                49 bits
t = 786433    → ceil(log₂(t)) = 20 bits
Max Budget    = 438 − 49 − 20 = 369 bits
```

MultDepth threshold: 369 / 33.38 ≈ **11.1** — constrains only expressions with multiplicative depth ≥ 12.

This is the same parameter configuration used to calibrate our `NoiseEstimator` (trained on data measured with a 369-bit system). Including it as a training budget ensures the agent encounters the full-capacity scenario during training, and serves as a near-unconstrained reference within the real SEAL parameter space.

### Budget 5: 9,000,000 (Unconstrained Control)

Sentinel value representing "no noise constraint." Used as the control condition to measure pure cost optimization performance without any budget pressure. No real SEAL configuration produces this budget — it is a training artifact to ensure the agent can still optimize cost when no noise constraint is active.

### Summary Table

| # | Budget (bits) | `poly_modulus_degree` | Security | Total Q | Last Prime | `plain_modulus` | t bits | Formula | MultDepth Threshold |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **172** | 16384 | 256-bit | 237 | 48 | 65537 | 17 | 237−48−17 | ~5.2 (depth ≥ 6) |
| 2 | **230** | 16384 | 192-bit | 300 | 50 | 786433 | 20 | 300−50−20 | ~6.9 (depth ≥ 7) |
| 3 | **236** | 16384 | 192-bit | 300 | 50 | 12289 | 14 | 300−50−14 | ~7.1 (depth ≥ 8) |
| 4 | **369** | 16384 | 128-bit | 438 | 49 | 786433 | 20 | 438−49−20 | ~11.1 (depth ≥ 12) |
| 5 | **9,000,000** | — | — | — | — | — | — | Unconstrained | — |

### Design Rationale

The 4 constrained budgets span three security levels (128-bit, 192-bit, 256-bit) and two `plain_modulus` values (65537, 786433, 12289), covering the realistic range of SEAL BFV deployments:

- **172** represents a high-security (256-bit), bandwidth-constrained deployment where circuit depth is limited to ~5 multiplications.
- **230** and **236** represent 192-bit security deployments with different plaintext precision requirements (20-bit vs 14-bit), offering ~7 levels of multiplicative depth.
- **369** represents the maximum-capacity 128-bit deployment (our calibration system), allowing ~11 multiplications before noise exhaustion.

All values are in the same SEAL noise-budget bit unit system as our `NoiseEstimator`.

---

## 11. Summary

| Question | Answer |
|---|---|
| What is the noise budget? | A measure (in bits) of remaining capacity for homomorphic operations |
| What happens at 0 budget? | **Silent failure** — decryption returns wrong results, no error/exception |
| Can you recover (bootstrap)? | **No** — BFV in SEAL is leveled, no bootstrapping support |
| Typical fresh budget range? | **50–200 bits** for standard parameters (n=8192, 128-bit security) |
| What determines the budget? | `coeff_modulus`, `plain_modulus`, `poly_modulus_degree` |
| Most expensive operation? | Multiplication (~30–35 bits consumed per multiply) |
| Cheapest operation? | Addition (~0–1 bit), rotation (~3 bits in SEAL ≥ 3.3) |

---

## Sources

1. **SEAL GitHub Repository**: https://github.com/microsoft/SEAL
2. **SEAL BFV Basics Example** (`1_bfv_basics.cpp`): https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp
3. **SEAL Default Parameters Source** (`globals.cpp`): https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/globals.cpp
4. **SEAL Issue #226 — Noise Budget Questions**: https://github.com/microsoft/SEAL/issues/226
5. **StackOverflow — Decryption Error Indicators**: https://stackoverflow.com/questions/54962785/decoding-and-decryption-error-indicators-in-seal
6. **StackOverflow — Noise Budget Calculation**: https://crypto.stackexchange.com/questions/107926/understanding-noise-budget-calculation-in-seal
7. **StackOverflow — Choosing SEAL Parameters**: https://stackoverflow.com/questions/52150812/choosing-the-suitable-parameters-for-seal-and-clarifying-some-mathematical-opera
8. **Pyfhel BFV Integer Demo**: https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_2_Integer_BFV.html
9. **SEAL Manual v2.3.1**: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/sealmanual-2-3-1.pdf
10. **Guide to SEAL**: https://magic3007.github.io/blog/2020/05/12/A-Guide-to-Homomorphic-Encryption-Library-SEAL.html
11. **SEAL Depth Estimator**: https://github.com/WeiDaiWD/SEAL-Depth-Estimator
12. **FHE Textbook — BFV Scheme**: https://fhetextbook.github.io/BFVScheme.html
13. **Relaxed Functional Bootstrapping (ePrint 2024/172)**: https://eprint.iacr.org/2024/172
14. **Noise Growth in RNS BFV (ePrint 2019/1266)**: https://eprint.iacr.org/2019/1266
15. **Improved BFV Noise Bound (ePrint 2025/899)**: https://eprint.iacr.org/2025/899
