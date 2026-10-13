/**
 * Executes a CHEHAB program trace (see make_trace.py) on HEonGPU and reports
 * memory read from the RMM statistics adaptor of HEonGPU's memory pool.
 *
 *   gpu_trace_probe <program.trace> <name>
 */
#include <heongpu/heongpu.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

constexpr auto CKKS = heongpu::Scheme::CKKS;
using StatsMR = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
using Ct = heongpu::Ciphertext<CKKS>;

static StatsMR* stats_mr()
{
    return dynamic_cast<StatsMR*>(heongpu::MemoryPool::instance().get_device_resource());
}
static long long cur_bytes() { cudaDeviceSynchronize(); return stats_mr()->get_bytes_counter().value; }
static long long peak_bytes() { cudaDeviceSynchronize(); return stats_mr()->get_bytes_counter().peak; }

static std::vector<int> parse_list(const std::string& s)
{
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
    return v;
}

int main(int argc, char** argv)
{
    std::ifstream in(argv[1]);
    const std::string name = argv[2];
    std::string kw, q_s, p_s;
    int logN;
    in >> kw >> logN >> q_s >> p_s;
    std::vector<int> q_bits = parse_list(q_s), p_bits = parse_list(p_s);
    // HEonGPU requires every P prime to be at least as large as the largest Q prime.
    // Each RNS limb is stored in a 64-bit word, so this does not change memory.
    const int max_q = *std::max_element(q_bits.begin(), q_bits.end());
    for (int& p : p_bits) p = std::max(p, max_q);
    const size_t N = size_t(1) << logN;
    const double scale = std::pow(2.0, 40);

    heongpu::HEContext<CKKS> ctx = heongpu::GenHEContext<CKKS>();
    ctx->set_poly_modulus_degree(N);
    ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    ctx->generate();
    if (!stats_mr()) { std::fprintf(stderr, "RMM statistics adaptor not available\n"); return 1; }
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

    heongpu::HEEncoder<CKKS> encoder(ctx);
    heongpu::HEEncryptor<CKKS> encryptor(ctx, pk);
    heongpu::HEArithmeticOperator<CKKS> ops(ctx, encoder);
    const long long m_tools = cur_bytes();

    std::vector<double> data(N / 2, 0.5);
    std::unordered_map<int, Ct> vals;
    long long m_inputs = -1;
    int n_ops = 0;

    std::string op;
    while (in >> op) {
        if (op == "IN") {
            int id; in >> id;
            heongpu::Plaintext<CKKS> pt(ctx);
            encoder.encode(pt, data, scale);
            Ct ct(ctx);
            encryptor.encrypt(ct, pt);
            vals.emplace(id, std::move(ct));
            continue;
        }
        if (m_inputs < 0) m_inputs = cur_bytes();
        if (op == "MUL" || op == "ADD") {
            int d, a, b; in >> d >> a >> b;
            // The result is a new ciphertext (like Lattigo's *New ops); the second
            // operand is only copied when its level has to be lowered.
            Ct x(vals.at(a));
            while (x.depth() < vals.at(b).depth()) ops.mod_drop_inplace(x);
            Ct* y = &vals.at(b);
            Ct y_low;
            if (y->depth() < x.depth()) {
                y_low = *y;
                while (y_low.depth() < x.depth()) ops.mod_drop_inplace(y_low);
                y = &y_low;
            }
            if (op == "MUL") {
                ops.multiply_inplace(x, *y);
                ops.relinearize_inplace(x, rlk);
            } else {
                ops.add_inplace(x, *y);
            }
            vals.insert_or_assign(d, std::move(x));
            ++n_ops;
        } else if (op == "RS") {
            int id; in >> id;
            ops.rescale_inplace(vals.at(id));
            ++n_ops;
        } else if (op == "FREE") {
            int id; in >> id;
            vals.erase(id);
        } else if (op == "OUT") {
            int id; in >> id;
        }
    }
    const long long m_end = cur_bytes();
    const long long m_peak = peak_bytes();

    // name,N,L,K,ops,ctx,skpk,rlk,tools,after_inputs,end,peak
    std::printf("%s,%zu,%zu,%zu,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                name.c_str(), N, q_bits.size(), p_bits.size(), n_ops,
                m_ctx, m_skpk - m_ctx, m_rlk - m_skpk, m_tools - m_rlk,
                m_inputs, m_end, m_peak);
    return 0;
}
