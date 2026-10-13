/**
 * GPU memory probe (HEonGPU CKKS).
 *
 * Runs one configuration per process so the RMM peak counter is clean:
 *   gpu_mem_probe <N> <num_q_primes> <num_rot_keys> <live_cts> <rep> [num_p_primes=1]
 *
 * Q chain = {60, 40 x (L-1)}, P = {60 x K}, scale = 2^40.
 * Galois keys are generated only for shifts {1..k} (explicit shift vector).
 * Workload: encrypt w ciphertexts, acc = ct0 + sum_i rot(ct[i%w], i),
 *           acc = acc * ct0, relinearize, rescale.
 *
 * Memory is read from HEonGPU's RMM statistics adaptor (bytes actually
 * allocated inside the pool), not from cudaMemGetInfo, because the pool
 * reserves device memory up front.
 */
#include <heongpu/heongpu.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr auto CKKS = heongpu::Scheme::CKKS;
using StatsMR = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

static StatsMR* stats_mr()
{
    return dynamic_cast<StatsMR*>(heongpu::MemoryPool::instance().get_device_resource());
}

static long long cur_bytes()
{
    cudaDeviceSynchronize();
    if (auto* s = stats_mr()) return s->get_bytes_counter().value;
    return (long long)heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();
}

static long long peak_bytes()
{
    cudaDeviceSynchronize();
    if (auto* s = stats_mr()) return s->get_bytes_counter().peak;
    return -1;
}

static long long driver_used_bytes()
{
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return (long long)(total_b - free_b);
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s N L k w rep\n", argv[0]);
        return 1;
    }
    const size_t N = std::atol(argv[1]);
    const int L = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);
    const int w = std::atoi(argv[4]);
    const int rep = std::atoi(argv[5]);
    const int K = argc > 6 ? std::atoi(argv[6]) : 1;

    std::vector<int> q_bits{60};
    for (int i = 1; i < L; ++i) q_bits.push_back(40);
    std::vector<int> p_bits(K, 60);
    const double scale = std::pow(2.0, 40);
    const int slots = N / 2;

    const long long drv0 = driver_used_bytes();

    heongpu::HEContext<CKKS> ctx = heongpu::GenHEContext<CKKS>();
    ctx->set_poly_modulus_degree(N);
    ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    ctx->generate();
    const long long m_ctx = cur_bytes();

    heongpu::HEKeyGenerator<CKKS> keygen(ctx);
    heongpu::Secretkey<CKKS> sk(ctx);
    keygen.generate_secret_key(sk);
    heongpu::Publickey<CKKS> pk(ctx);
    keygen.generate_public_key(pk, sk);
    const long long m_skpk = cur_bytes();

    heongpu::Relinkey<CKKS> rlk(ctx);
    keygen.generate_relin_key(rlk, sk);
    const long long m_rlk = cur_bytes();

    std::vector<int> shifts;
    for (int i = 1; i <= k; ++i) shifts.push_back(i);
    heongpu::Galoiskey<CKKS> glk = k > 0 ? heongpu::Galoiskey<CKKS>(ctx, shifts)
                                         : heongpu::Galoiskey<CKKS>();
    if (k > 0) keygen.generate_galois_key(glk, sk);
    const long long m_gk = cur_bytes();

    heongpu::HEEncoder<CKKS> encoder(ctx);
    heongpu::HEEncryptor<CKKS> encryptor(ctx, pk);
    heongpu::HEArithmeticOperator<CKKS> ops(ctx, encoder);
    const long long m_tools = cur_bytes();

    std::vector<double> data(slots, 0.5);
    std::vector<heongpu::Ciphertext<CKKS>> cts;
    cts.reserve(w);
    for (int j = 0; j < w; ++j) {
        heongpu::Plaintext<CKKS> pt(ctx);
        encoder.encode(pt, data, scale);
        cts.emplace_back(ctx);
        encryptor.encrypt(cts.back(), pt);
    }
    const long long m_cts = cur_bytes();

    heongpu::Ciphertext<CKKS> acc(cts[0]);
    for (int i = 1; i <= k; ++i) {
        heongpu::Ciphertext<CKKS> tmp(cts[i % w]);
        ops.rotate_rows_inplace(tmp, glk, i);
        ops.add_inplace(acc, tmp);
    }
    ops.multiply_inplace(acc, cts[0]);
    ops.relinearize_inplace(acc, rlk);
    ops.rescale_inplace(acc);
    const long long m_end = cur_bytes();
    const long long m_peak = peak_bytes();
    const long long drv1 = driver_used_bytes();

    // backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak,driver_used_delta
    std::printf("gpu,%zu,%d,%zu,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                N, L, p_bits.size(), k, w, rep,
                m_ctx, m_skpk - m_ctx, m_rlk - m_skpk, m_gk - m_rlk,
                m_tools - m_gk, m_cts - m_tools, m_end, m_peak, drv1 - drv0);
    return 0;
}
