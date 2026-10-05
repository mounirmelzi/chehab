#include "fheco/ckks/ckks_params.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace fheco::ckks
{

// Security standards from Homomorphic Encryption Standard
// Format: {log_n, max_total_log_q}
const std::vector<std::pair<int, int>> CKKSParamSelector::security_standard_128 = {
  {10, 27},   // n=1024
  {11, 54},   // n=2048
  {12, 109},  // n=4096
  {13, 218},  // n=8192
  {14, 438},  // n=16384
  {15, 881},  // n=32768
  {16, 1770}  // n=65536
};

const std::vector<std::pair<int, int>> CKKSParamSelector::security_standard_192 = {
  {10, 19},
  {11, 37},
  {12, 75},
  {13, 152},
  {14, 305},
  {15, 611},
  {16, 1228}
};

const std::vector<std::pair<int, int>> CKKSParamSelector::security_standard_256 = {
  {10, 14},
  {11, 29},
  {12, 58},
  {13, 118},
  {14, 237},
  {15, 476},
  {16, 956}
};

int CKKSParamSelector::get_max_log_q(int log_n, SecurityLevel sec_level)
{
  const std::vector<std::pair<int, int>>* standard = nullptr;
  
  switch (sec_level)
  {
    case SecurityLevel::tc128: standard = &security_standard_128; break;
    case SecurityLevel::tc192: standard = &security_standard_192; break;
    case SecurityLevel::tc256: standard = &security_standard_256; break;
  }
  
  for (const auto& entry : *standard)
  {
    if (entry.first == log_n)
      return entry.second;
  }
  
  // If log_n not found, return conservative estimate
  if (log_n < 10) return 0;
  if (log_n > 16) return security_standard_128.back().second * 2; // Extrapolate
  
  return 0;
}

CKKSParams CKKSParamSelector::select_params(
  int mult_depth,
  double precision,
  std::size_t min_slots,
  SecurityLevel sec_level)
{
  CKKSParams params;
  
  // 1. Determine scale from precision requirement
  // precision = 1/scale after mult_depth multiplications
  // We want final precision, so scale should be larger
  // Typical: log_scale = 40 for 1e-6 precision with depth ~10
  params.log_scale = std::max(30, static_cast<int>(std::ceil(-std::log2(precision))) + 10);
  params.log_scale = std::min(params.log_scale, 60); // Cap at 60 bits
  
  // 2. Build modulus chain
  // First prime (q_0) should be larger to accommodate initial noise
  // Each subsequent prime is for one multiplication level
  int first_prime_bits = params.log_scale + 10; // Extra bits for noise
  first_prime_bits = std::min(first_prime_bits, 60);
  
  params.log_q.push_back(first_prime_bits);
  for (int i = 0; i < mult_depth; ++i)
  {
    params.log_q.push_back(params.log_scale);
  }
  
  // 3. Special primes for key-switching
  // Typically 2-3 primes of size ~45-50 bits
  params.log_p = {45, 45};
  
  // 4. Determine ring dimension (log_n)
  // Must satisfy:
  // - slot_count >= min_slots (slot_count = n/2)
  // - total_log_q <= max_log_q for security
  
  int min_log_n = static_cast<int>(std::ceil(std::log2(min_slots * 2)));
  min_log_n = std::max(min_log_n, 10); // Minimum n = 1024
  
  int total_bits = params.total_log_q();
  for (int bits : params.log_p) total_bits += bits;
  
  // Find smallest log_n that satisfies security requirement
  params.log_n = min_log_n;
  while (params.log_n <= 16)
  {
    int max_bits = get_max_log_q(params.log_n, sec_level);
    if (total_bits <= max_bits)
      break;
    ++params.log_n;
  }
  
  if (params.log_n > 16)
  {
    throw std::runtime_error(
      "Cannot find secure parameters for mult_depth=" + std::to_string(mult_depth) +
      ". Total modulus bits (" + std::to_string(total_bits) + ") too large.");
  }
  
  return params;
}

CKKSParams CKKSParamSelector::default_params(int mult_depth)
{
  CKKSParams params;
  
  // Default scale
  params.log_scale = 40;
  
  // Build modulus chain
  params.log_q.push_back(55); // First prime (larger for noise margin)
  for (int i = 0; i < mult_depth; ++i)
  {
    params.log_q.push_back(40); // One prime per level
  }
  
  // Special primes for key-switching
  params.log_p = {45, 45};
  
  // Bootstrap primes (used if bootstrapping is enabled)
  // Based on Lattigo's default bootstrap parameters
  params.log_p_boot = {61, 61, 61, 61}; // 4 primes of 61 bits each
  params.enable_bootstrap = false; // Disabled by default
  
  // Choose log_n based on depth
  // More depth = more primes = need larger n for security
  int total_bits = params.total_log_q() + 90; // +90 for log_p
  
  if (total_bits <= 109) params.log_n = 12;       // n=4096
  else if (total_bits <= 218) params.log_n = 13;  // n=8192
  else if (total_bits <= 438) params.log_n = 14;  // n=16384
  else if (total_bits <= 881) params.log_n = 15;  // n=32768
  else params.log_n = 16;                          // n=65536
  
  return params;
}

CKKSParams CKKSParamSelector::default_params_with_bootstrap(int mult_depth)
{
  CKKSParams params;
  
  // For bootstrapping, Lattigo requires LogN=16 for efficient bootstrap circuit
  // The bootstrap circuit itself has depth ~15
  params.log_n = 16;
  
  // Default scale
  params.log_scale = 40;
  
  // Build modulus chain
  // First prime (larger for noise margin)
  params.log_q.push_back(55);
  for (int i = 0; i < mult_depth; ++i)
  {
    params.log_q.push_back(40); // One prime per level
  }
  
  // Special primes for key-switching
  params.log_p = {45, 45};
  
  // Bootstrap primes - Orion uses 61-bit primes
  // Number of primes: max(1, ceil(sqrt(#Qi))) typically 4-8
  int num_boot_primes = std::max(4, static_cast<int>(std::ceil(std::sqrt(params.log_q.size()))));
  params.log_p_boot.clear();
  for (int i = 0; i < num_boot_primes; ++i)
  {
    params.log_p_boot.push_back(61);
  }
  
  // Orion-compatible bootstrap parameters
  params.enable_bootstrap = true;
  params.hamming_weight = 192;  // Standard Hamming weight for security
  params.ring_type = RingType::Standard;  // Standard ring (use ConjugateInvariant for efficiency)
  params.log_slots = params.log_n - 1;    // n/2 slots (max for standard ring)
  
  return params;
}

bool CKKSParamSelector::validate_security(const CKKSParams& params, SecurityLevel sec_level)
{
  int total_bits = params.total_log_q();
  for (int bits : params.log_p) total_bits += bits;
  
  int max_bits = get_max_log_q(params.log_n, sec_level);
  return total_bits <= max_bits;
}

std::ostream& operator<<(std::ostream& os, const CKKSParams& params)
{
  os << "CKKSParams {\n";
  os << "  log_n: " << params.log_n << " (n=" << params.poly_modulus_degree() << ")\n";
  os << "  slot_count: " << params.slot_count() << "\n";
  os << "  log_q: [";
  for (size_t i = 0; i < params.log_q.size(); ++i)
  {
    if (i > 0) os << ", ";
    os << params.log_q[i];
  }
  os << "] (total=" << params.total_log_q() << " bits)\n";
  os << "  log_p: [";
  for (size_t i = 0; i < params.log_p.size(); ++i)
  {
    if (i > 0) os << ", ";
    os << params.log_p[i];
  }
  os << "]\n";
  os << "  log_scale: " << params.log_scale << " (scale=" << params.scale() << ")\n";
  os << "  max_level: " << params.max_level() << "\n";
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, CKKSParamSelector::SecurityLevel level)
{
  switch (level)
  {
    case CKKSParamSelector::SecurityLevel::tc128: os << "128-bit"; break;
    case CKKSParamSelector::SecurityLevel::tc192: os << "192-bit"; break;
    case CKKSParamSelector::SecurityLevel::tc256: os << "256-bit"; break;
  }
  return os;
}

} // namespace fheco::ckks

