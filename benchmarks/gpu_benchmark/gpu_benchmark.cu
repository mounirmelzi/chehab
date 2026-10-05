/**
 * Standalone HEonGPU benchmark — GPU-accelerated BFV/CKKS operations.
 *
 * Runs common FHE operations on GPU via HEonGPU and reports separated timing
 * (keygen, encrypt, eval, decrypt) to demonstrate GPU acceleration.
 */
#include <heongpu/heongpu.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct TimingResult {
    std::string name;
    double keygen_ms;
    double encrypt_ms;
    double eval_ms;
    double decrypt_ms;
    double total_ms;
    bool correct;
};

constexpr auto BFV = heongpu::Scheme::BFV;
constexpr auto CKKS = heongpu::Scheme::CKKS;

TimingResult run_bfv_benchmark(std::size_t poly_modulus_degree,
                                const std::vector<int>& q_bits,
                                const std::vector<int>& p_bits,
                                int plain_modulus = 1032193)
{
    TimingResult result;
    result.name = "BFV_N" + std::to_string(poly_modulus_degree);

    auto t0 = Clock::now();

    heongpu::HEContext<BFV> context = heongpu::GenHEContext<BFV>();
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    context->set_plain_modulus(plain_modulus);
    context->generate();

    heongpu::HEKeyGenerator<BFV> keygen(context);
    heongpu::Secretkey<BFV> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<BFV> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<BFV> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    cudaDeviceSynchronize();
    auto t_keygen = Clock::now();
    result.keygen_ms = std::chrono::duration_cast<Ms>(t_keygen - t0).count();

    heongpu::HEEncoder<BFV> encoder(context);
    heongpu::HEEncryptor<BFV> encryptor(context, public_key);
    heongpu::HEDecryptor<BFV> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<BFV> ops(context, encoder);

    std::vector<uint64_t> msg_a(poly_modulus_degree, 7ULL);
    std::vector<uint64_t> msg_b(poly_modulus_degree, 3ULL);

    heongpu::Plaintext<BFV> pt_a(context), pt_b(context);
    encoder.encode(pt_a, msg_a);
    encoder.encode(pt_b, msg_b);

    heongpu::Ciphertext<BFV> ct_a(context), ct_b(context);

    auto t_enc_start = Clock::now();
    encryptor.encrypt(ct_a, pt_a);
    encryptor.encrypt(ct_b, pt_b);
    cudaDeviceSynchronize();
    auto t_enc_end = Clock::now();
    result.encrypt_ms = std::chrono::duration_cast<Ms>(t_enc_end - t_enc_start).count();

    auto t_eval_start = Clock::now();
    ops.add_inplace(ct_a, ct_b);
    ops.multiply_inplace(ct_a, ct_b);
    ops.relinearize_inplace(ct_a, relin_key);
    cudaDeviceSynchronize();
    auto t_eval_end = Clock::now();
    result.eval_ms = std::chrono::duration_cast<Ms>(t_eval_end - t_eval_start).count();

    heongpu::Plaintext<BFV> pt_result(context);
    auto t_dec_start = Clock::now();
    decryptor.decrypt(pt_result, ct_a);
    cudaDeviceSynchronize();
    auto t_dec_end = Clock::now();
    result.decrypt_ms = std::chrono::duration_cast<Ms>(t_dec_end - t_dec_start).count();

    result.total_ms = result.keygen_ms + result.encrypt_ms + result.eval_ms + result.decrypt_ms;

    std::vector<uint64_t> decoded;
    encoder.decode(decoded, pt_result);
    uint64_t expected = (7 + 3) * 3;
    result.correct = (decoded[0] == expected);

    return result;
}

TimingResult run_ckks_benchmark(std::size_t poly_modulus_degree,
                                 const std::vector<int>& modulus_bits,
                                 const std::vector<int>& special_bits)
{
    TimingResult result;
    result.name = "CKKS_N" + std::to_string(poly_modulus_degree);

    auto t0 = Clock::now();

    heongpu::HEContext<CKKS> context = heongpu::GenHEContext<CKKS>();
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(modulus_bits, special_bits);
    context->generate();

    double scale = pow(2.0, 40);

    heongpu::HEKeyGenerator<CKKS> keygen(context);
    heongpu::Secretkey<CKKS> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    cudaDeviceSynchronize();
    auto t_keygen = Clock::now();
    result.keygen_ms = std::chrono::duration_cast<Ms>(t_keygen - t0).count();

    heongpu::HEEncoder<CKKS> encoder(context);
    heongpu::HEEncryptor<CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<CKKS> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<CKKS> ops(context, encoder);

    std::size_t slot_count = poly_modulus_degree / 2;
    std::vector<double> msg_a(slot_count, 3.14159);
    std::vector<double> msg_b(slot_count, 2.71828);

    heongpu::Plaintext<CKKS> pt_a(context), pt_b(context);
    encoder.encode(pt_a, msg_a, scale);
    encoder.encode(pt_b, msg_b, scale);

    heongpu::Ciphertext<CKKS> ct_a(context), ct_b(context);

    auto t_enc_start = Clock::now();
    encryptor.encrypt(ct_a, pt_a);
    encryptor.encrypt(ct_b, pt_b);
    cudaDeviceSynchronize();
    auto t_enc_end = Clock::now();
    result.encrypt_ms = std::chrono::duration_cast<Ms>(t_enc_end - t_enc_start).count();

    auto t_eval_start = Clock::now();
    ops.add_inplace(ct_a, ct_b);
    ops.multiply_inplace(ct_a, ct_b);
    ops.relinearize_inplace(ct_a, relin_key);
    ops.rescale_inplace(ct_a);
    cudaDeviceSynchronize();
    auto t_eval_end = Clock::now();
    result.eval_ms = std::chrono::duration_cast<Ms>(t_eval_end - t_eval_start).count();

    heongpu::Plaintext<CKKS> pt_result(context);
    auto t_dec_start = Clock::now();
    decryptor.decrypt(pt_result, ct_a);
    cudaDeviceSynchronize();
    auto t_dec_end = Clock::now();
    result.decrypt_ms = std::chrono::duration_cast<Ms>(t_dec_end - t_dec_start).count();

    result.total_ms = result.keygen_ms + result.encrypt_ms + result.eval_ms + result.decrypt_ms;

    std::vector<double> decoded;
    encoder.decode(decoded, pt_result);
    double expected = (3.14159 + 2.71828) * 2.71828;
    double rel_error = std::abs(decoded[0] - expected) / std::abs(expected);
    result.correct = (rel_error < 1e-3);

    std::cout << "  Decoded[0]=" << decoded[0]
              << "  Expected=" << expected
              << "  RelError=" << rel_error << std::endl;

    return result;
}

void print_results(const std::vector<TimingResult>& results)
{
    std::cout << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  GPU Benchmark Results (HEonGPU)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << std::left << std::setw(24) << "Benchmark"
              << std::right
              << std::setw(12) << "KeyGen(ms)"
              << std::setw(12) << "Encrypt(ms)"
              << std::setw(12) << "Eval(ms)"
              << std::setw(12) << "Decrypt(ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(10) << "Correct"
              << std::endl;
    std::cout << std::string(94, '-') << std::endl;

    for (const auto& r : results)
    {
        std::cout << std::left << std::setw(24) << r.name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << r.keygen_ms
                  << std::setw(12) << r.encrypt_ms
                  << std::setw(12) << r.eval_ms
                  << std::setw(12) << r.decrypt_ms
                  << std::setw(12) << r.total_ms
                  << std::setw(10) << (r.correct ? "YES" : "NO")
                  << std::endl;
    }
    std::cout << std::endl;

    std::cout << "CSV:" << std::endl;
    std::cout << "benchmark,keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,correct" << std::endl;
    for (const auto& r : results)
    {
        std::cout << r.name << ","
                  << r.keygen_ms << "," << r.encrypt_ms << ","
                  << r.eval_ms << "," << r.decrypt_ms << ","
                  << r.total_ms << "," << (r.correct ? 1 : 0)
                  << std::endl;
    }
}

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "ERROR: No CUDA-capable GPU found." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name
              << " (Compute " << prop.major << "." << prop.minor
              << ", " << prop.totalGlobalMem / (1024*1024) << " MB)"
              << std::endl;

    std::vector<TimingResult> results;

    // BFV benchmarks -- plain_modulus must satisfy p ≡ 1 mod 2N for batch NTT
    std::cout << std::endl << "Running BFV benchmarks..." << std::endl;
    try {
        std::cout << "  BFV N=8192 (plain_mod=1032193) ..." << std::endl;
        results.push_back(run_bfv_benchmark(8192, {40, 40, 40, 40}, {40}));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    try {
        // 786433 ≡ 1 mod 32768 (786432 = 12*65536)
        std::cout << "  BFV N=16384 (plain_mod=786433) ..." << std::endl;
        results.push_back(run_bfv_benchmark(16384, {40, 40, 40, 40, 40, 40, 40}, {40}, 786433));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    try {
        // 786433 ≡ 1 mod 65536
        std::cout << "  BFV N=32768 (plain_mod=786433) ..." << std::endl;
        results.push_back(run_bfv_benchmark(32768, {40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {40}, 786433));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }

    // CKKS benchmarks -- total modulus bits must fit sec128 for ring size
    std::cout << std::endl << "Running CKKS benchmarks..." << std::endl;
    try {
        // N=8192 sec128 allows ~218 bits: 60+30+30+30+60=210
        std::cout << "  CKKS N=8192 ..." << std::endl;
        results.push_back(run_ckks_benchmark(8192, {60, 30, 30, 30}, {60}));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    try {
        std::cout << "  CKKS N=16384 ..." << std::endl;
        results.push_back(run_ckks_benchmark(16384, {60, 40, 40, 40, 40, 40, 40, 40}, {60}));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    try {
        std::cout << "  CKKS N=32768 ..." << std::endl;
        results.push_back(run_ckks_benchmark(32768, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {60}));
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }

    if (!results.empty()) {
        print_results(results);
    } else {
        std::cerr << "All benchmarks failed." << std::endl;
        return 1;
    }

    return 0;
}
