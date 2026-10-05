/**
 * GPU DNN-operator benchmark — mirrors the CPU (Lattigo) evaluation in
 * Table 5.11 / 5.12 of the manuscript for direct CPU-vs-GPU comparison.
 *
 * All operations use CKKS on HEonGPU.
 */
#include <heongpu/heongpu.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

constexpr auto CKKS = heongpu::Scheme::CKKS;

struct DnnResult {
    std::string name;
    double keygen_ms;
    double encrypt_ms;
    double eval_ms;
    double decrypt_ms;
    double total_ms;
};

// ─────────────── DNN operation implementations ──────────────────

// poly_act_deg2: x → x²  (1 mul + relin + rescale)
void poly_act_deg2(heongpu::HEArithmeticOperator<CKKS>& ops,
                   heongpu::Ciphertext<CKKS>& ct,
                   heongpu::Relinkey<CKKS>& rlk)
{
    ops.multiply_inplace(ct, ct);
    ops.relinearize_inplace(ct, rlk);
    ops.rescale_inplace(ct);
}

// poly_act_deg4: x → x⁴  (2× square + relin + rescale)
void poly_act_deg4(heongpu::HEArithmeticOperator<CKKS>& ops,
                   heongpu::Ciphertext<CKKS>& ct,
                   heongpu::Relinkey<CKKS>& rlk)
{
    ops.multiply_inplace(ct, ct);
    ops.relinearize_inplace(ct, rlk);
    ops.rescale_inplace(ct);
    ops.multiply_inplace(ct, ct);
    ops.relinearize_inplace(ct, rlk);
    ops.rescale_inplace(ct);
}

// drop plaintext levels to match ciphertext after previous rescales
void drop_plain(heongpu::HEArithmeticOperator<CKKS>& ops,
                heongpu::Plaintext<CKKS>& pt, int levels)
{
    for (int i = 0; i < levels; ++i)
        ops.mod_drop_inplace(pt);
}

// conv_like: k×k filter — rotate + mul_plain + accumulate
void conv_like(heongpu::HEArithmeticOperator<CKKS>& ops,
               heongpu::HEEncoder<CKKS>& encoder,
               heongpu::HEContext<CKKS>& context,
               heongpu::Ciphertext<CKKS>& ct,
               heongpu::Galoiskey<CKKS>& glk,
               int kernel_size, double scale, int slots,
               int levels_down = 0)
{
    int k2 = kernel_size * kernel_size;

    std::vector<heongpu::Plaintext<CKKS>> weights;
    for (int i = 0; i < k2; ++i) {
        std::vector<double> w(slots, 0.1 * (i + 1));
        weights.emplace_back(context);
        encoder.encode(weights.back(), w, scale);
        drop_plain(ops, weights.back(), levels_down);
    }

    heongpu::Ciphertext<CKKS> acc(ct);
    ops.rotate_rows_inplace(acc, glk, 1);
    ops.multiply_plain_inplace(acc, weights[0]);

    for (int i = 1; i < k2; ++i) {
        heongpu::Ciphertext<CKKS> tmp(ct);
        int step = i + 1;
        ops.rotate_rows_inplace(tmp, glk, step);
        ops.multiply_plain_inplace(tmp, weights[i]);
        ops.add_inplace(acc, tmp);
    }
    ops.rescale_inplace(acc);
    ct = acc;
}

// fc_like: fully connected layer — sum of rotate+mul_plain
void fc_like(heongpu::HEArithmeticOperator<CKKS>& ops,
             heongpu::HEEncoder<CKKS>& encoder,
             heongpu::HEContext<CKKS>& context,
             heongpu::Ciphertext<CKKS>& ct,
             heongpu::Galoiskey<CKKS>& glk,
             int dim, double scale, int slots,
             int levels_down = 0)
{
    std::vector<heongpu::Plaintext<CKKS>> weights;
    for (int i = 0; i < dim; ++i) {
        std::vector<double> w(slots, 0.1 * (i + 1));
        weights.emplace_back(context);
        encoder.encode(weights.back(), w, scale);
        drop_plain(ops, weights.back(), levels_down);
    }

    heongpu::Ciphertext<CKKS> acc(ct);
    ops.multiply_plain_inplace(acc, weights[0]);

    for (int i = 1; i < dim; ++i) {
        heongpu::Ciphertext<CKKS> tmp(ct);
        ops.rotate_rows_inplace(tmp, glk, i);
        ops.multiply_plain_inplace(tmp, weights[i]);
        ops.add_inplace(acc, tmp);
    }
    ops.rescale_inplace(acc);
    ct = acc;
}

// ─────────────────────── main ───────────────────────────────────

void print_results(const std::vector<DnnResult>& results)
{
    std::cout << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  GPU DNN-Operator Benchmark (HEonGPU CKKS)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << std::left  << std::setw(28) << "Operation"
              << std::right
              << std::setw(12) << "KeyGen(ms)"
              << std::setw(12) << "Encrypt(ms)"
              << std::setw(12) << "Eval(ms)"
              << std::setw(12) << "Decrypt(ms)"
              << std::setw(12) << "Total(ms)"
              << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    for (const auto& r : results) {
        std::cout << std::left  << std::setw(28) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.keygen_ms
                  << std::setw(12) << r.encrypt_ms
                  << std::setw(12) << r.eval_ms
                  << std::setw(12) << r.decrypt_ms
                  << std::setw(12) << r.total_ms
                  << std::endl;
    }

    std::cout << std::endl << "CSV:" << std::endl;
    std::cout << "operation,keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms" << std::endl;
    for (const auto& r : results) {
        std::cout << r.name
                  << "," << r.keygen_ms << "," << r.encrypt_ms
                  << "," << r.eval_ms   << "," << r.decrypt_ms
                  << "," << r.total_ms  << std::endl;
    }
}

DnnResult run_bench(const std::string& name,
                    std::size_t N,
                    const std::vector<int>& q_bits,
                    const std::vector<int>& p_bits,
                    double scale_pow,
                    bool need_galois,
                    std::function<void(
                        heongpu::HEArithmeticOperator<CKKS>&,
                        heongpu::HEEncoder<CKKS>&,
                        heongpu::HEContext<CKKS>&,
                        heongpu::Ciphertext<CKKS>&,
                        heongpu::Relinkey<CKKS>&,
                        heongpu::Galoiskey<CKKS>&,
                        double, int)> eval_fn)
{
    DnnResult r;
    r.name = name;

    double scale = pow(2.0, scale_pow);
    int slots = N / 2;

    // ── keygen ──
    auto t0 = Clock::now();
    heongpu::HEContext<CKKS> ctx = heongpu::GenHEContext<CKKS>();
    ctx->set_poly_modulus_degree(N);
    ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    ctx->generate();

    heongpu::HEKeyGenerator<CKKS> keygen(ctx);
    heongpu::Secretkey<CKKS> sk(ctx);
    keygen.generate_secret_key(sk);
    heongpu::Publickey<CKKS> pk(ctx);
    keygen.generate_public_key(pk, sk);
    heongpu::Relinkey<CKKS> rlk(ctx);
    keygen.generate_relin_key(rlk, sk);
    heongpu::Galoiskey<CKKS> glk(ctx);
    if (need_galois) {
        keygen.generate_galois_key(glk, sk);
    }
    cudaDeviceSynchronize();
    auto t1 = Clock::now();
    r.keygen_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();

    heongpu::HEEncoder<CKKS> encoder(ctx);
    heongpu::HEEncryptor<CKKS> encryptor(ctx, pk);
    heongpu::HEDecryptor<CKKS> decryptor(ctx, sk);
    heongpu::HEArithmeticOperator<CKKS> ops(ctx, encoder);

    // ── encrypt ──
    std::vector<double> data(slots, 0.5);
    heongpu::Plaintext<CKKS> pt(ctx);
    encoder.encode(pt, data, scale);
    heongpu::Ciphertext<CKKS> ct(ctx);

    auto t2 = Clock::now();
    encryptor.encrypt(ct, pt);
    cudaDeviceSynchronize();
    auto t3 = Clock::now();
    r.encrypt_ms = std::chrono::duration_cast<Ms>(t3 - t2).count();

    // ── eval ──
    auto t4 = Clock::now();
    eval_fn(ops, encoder, ctx, ct, rlk, glk, scale, slots);
    cudaDeviceSynchronize();
    auto t5 = Clock::now();
    r.eval_ms = std::chrono::duration_cast<Ms>(t5 - t4).count();

    // ── decrypt ──
    heongpu::Plaintext<CKKS> pt_out(ctx);
    auto t6 = Clock::now();
    decryptor.decrypt(pt_out, ct);
    cudaDeviceSynchronize();
    auto t7 = Clock::now();
    r.decrypt_ms = std::chrono::duration_cast<Ms>(t7 - t6).count();

    r.total_ms = r.keygen_ms + r.encrypt_ms + r.eval_ms + r.decrypt_ms;
    return r;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name
              << " (Compute " << prop.major << "." << prop.minor
              << ", " << prop.totalGlobalMem / (1024*1024) << " MB)" << std::endl;

    std::vector<DnnResult> results;

    // Matching Table 5.12 from manuscript (CPU Lattigo timings)
    // All use N=16384, scale=2^40

    // ── Polynomial activations ──
    std::cout << std::endl << "=== Polynomial activations ===" << std::endl;

    try {
        std::cout << "  poly_act_deg2 ..." << std::endl;
        results.push_back(run_bench("poly_act_deg2", 16384,
            {60, 40, 40, 40}, {60}, 40.0, false,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                poly_act_deg2(ops, ct, rlk);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    try {
        std::cout << "  poly_act_deg4 ..." << std::endl;
        results.push_back(run_bench("poly_act_deg4", 16384,
            {60, 40, 40, 40, 40, 40}, {60}, 40.0, false,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                poly_act_deg4(ops, ct, rlk);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    // ── Convolutions ──
    std::cout << std::endl << "=== Convolution-like ===" << std::endl;

    try {
        std::cout << "  conv2d_5x5_k3 ..." << std::endl;
        results.push_back(run_bench("conv2d_5x5_k3", 16384,
            {60, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                conv_like(ops, enc, ctx, ct, glk, 3, s, sl);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    try {
        std::cout << "  conv2d_7x7_k3 ..." << std::endl;
        results.push_back(run_bench("conv2d_7x7_k3", 16384,
            {60, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                conv_like(ops, enc, ctx, ct, glk, 3, s, sl);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    // ── FC layers ──
    std::cout << std::endl << "=== Fully connected layers ===" << std::endl;

    try {
        std::cout << "  fc_plus_act_4 ..." << std::endl;
        results.push_back(run_bench("fc_plus_act_4", 16384,
            {60, 40, 40, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl);
                poly_act_deg2(ops, ct, rlk);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    try {
        std::cout << "  fc_plus_act_8 ..." << std::endl;
        results.push_back(run_bench("fc_plus_act_8", 16384,
            {60, 40, 40, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 8, s, sl);
                poly_act_deg2(ops, ct, rlk);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    // ── MatMul ──
    std::cout << std::endl << "=== Matrix multiply ===" << std::endl;

    try {
        std::cout << "  matmul_4x4 ..." << std::endl;
        results.push_back(run_bench("matmul_4x4", 16384,
            {60, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    // ── FC 2-layer ──
    try {
        std::cout << "  fc_2layer_4 ..." << std::endl;
        results.push_back(run_bench("fc_2layer_4", 16384,
            {60, 40, 40, 40, 40, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl, 0);
                poly_act_deg2(ops, ct, rlk);         // +2 rescales total
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl, 2);
                poly_act_deg2(ops, ct, rlk);         // +4 rescales total
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    // ── Larger workloads (matching Table 5.11) ──
    std::cout << std::endl << "=== Larger workloads (Table 5.11) ===" << std::endl;

    try {
        std::cout << "  conv_8x8_k3_1layer ..." << std::endl;
        results.push_back(run_bench("conv_8x8_k3_1layer", 16384,
            {60, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                conv_like(ops, enc, ctx, ct, glk, 3, s, sl);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    try {
        std::cout << "  linear_4x4_2layer ..." << std::endl;
        results.push_back(run_bench("linear_4x4_2layer", 16384,
            {60, 40, 40, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl, 0);
                fc_like(ops, enc, ctx, ct, glk, 4, s, sl, 1);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    try {
        std::cout << "  linear_8x8_2layer ..." << std::endl;
        results.push_back(run_bench("linear_8x8_2layer", 16384,
            {60, 40, 40, 40, 40, 40}, {60}, 40.0, true,
            [](auto& ops, auto& enc, auto& ctx, auto& ct, auto& rlk, auto& glk, double s, int sl) {
                fc_like(ops, enc, ctx, ct, glk, 8, s, sl, 0);
                fc_like(ops, enc, ctx, ct, glk, 8, s, sl, 1);
            }));
    } catch (const std::exception& e) { std::cerr << "  FAILED: " << e.what() << std::endl; }

    if (!results.empty()) {
        print_results(results);
    } else {
        std::cerr << "All DNN benchmarks failed." << std::endl;
        return 1;
    }

    return 0;
}
