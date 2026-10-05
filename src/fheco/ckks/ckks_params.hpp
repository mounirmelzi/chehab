#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <ostream>

namespace fheco::ckks
{

/**
 * Ring type for CKKS
 */
enum class RingType
{
  Standard,           // Standard ring (n complex slots)
  ConjugateInvariant  // Conjugate invariant ring (n real slots, more efficient)
};

/**
 * CKKS encryption parameters
 * 
 * In CKKS:
 * - n (poly_modulus_degree): Ring dimension, determines security and slot count
 * - Q = q_0 * q_1 * ... * q_L: Modulus chain, L+1 primes
 * - P = p_0 * p_1 * ...: Special primes for key-switching
 * - scale: Encoding precision (typically 2^40)
 * - H: Hamming weight of secret key (security parameter)
 * - RingType: Standard or ConjugateInvariant
 * 
 * Key relationships:
 * - slot_count = n/2 (for CKKS standard) or n (for ConjugateInvariant)
 * - max_level = L = number of primes in Q - 1
 * - Each multiplication consumes one level (after rescale)
 */
struct CKKSParams
{
  std::size_t log_n;              // log2(poly_modulus_degree)
  std::vector<int> log_q;         // Bit sizes of modulus chain primes
  std::vector<int> log_p;         // Bit sizes of special primes (for key-switching)
  std::vector<int> log_p_boot;    // Bit sizes of bootstrap special primes
  int log_scale;                  // log2(scale), typically 40
  bool enable_bootstrap = false;  // Whether bootstrapping is enabled
  
  // Additional parameters for Orion compatibility
  int hamming_weight = 192;       // Secret key Hamming weight (H), default 192 for security
  RingType ring_type = RingType::Standard;  // Ring type
  int log_slots = -1;             // log2(number of slots), -1 means use max
  
  // Computed values
  std::size_t poly_modulus_degree() const { return 1ULL << log_n; }
  
  std::size_t slot_count() const { 
    if (log_slots > 0) return 1ULL << log_slots;
    // ConjugateInvariant doubles effective slots
    return poly_modulus_degree() / 2;
  }
  
  std::size_t max_level() const { return log_q.size() - 1; }
  double scale() const { return static_cast<double>(1ULL << log_scale); }
  
  // Total modulus bit size (for security estimation)
  int total_log_q() const {
    int sum = 0;
    for (int bits : log_q) sum += bits;
    return sum;
  }
  
  // Get LogP for bootstrapping (use boot primes if available, else regular primes)
  const std::vector<int>& bootstrap_log_p() const {
    return log_p_boot.empty() ? log_p : log_p_boot;
  }
  
  // Get effective log slots for bootstrapping
  int effective_log_slots() const {
    if (log_slots > 0) return log_slots;
    return static_cast<int>(log_n) - 1;  // Default: n/2 slots
  }
};

/**
 * CKKS Parameter Selector
 * 
 * Selects appropriate CKKS parameters based on:
 * - Required multiplicative depth
 * - Required precision
 * - Required number of slots
 * - Security level
 */
class CKKSParamSelector
{
public:
  enum class SecurityLevel
  {
    tc128,  // 128-bit security
    tc192,  // 192-bit security
    tc256   // 256-bit security
  };
  
  /**
   * Select CKKS parameters
   * 
   * @param mult_depth Maximum multiplicative depth of the circuit
   * @param precision Required precision (e.g., 1e-6)
   * @param min_slots Minimum number of slots required
   * @param sec_level Security level
   * @return Selected CKKS parameters
   */
  static CKKSParams select_params(
    int mult_depth,
    double precision = 1e-6,
    std::size_t min_slots = 1,
    SecurityLevel sec_level = SecurityLevel::tc128);
  
  /**
   * Get default parameters for a given multiplicative depth
   * Uses conservative defaults
   */
  static CKKSParams default_params(int mult_depth);
  
  /**
   * Get default parameters with bootstrapping enabled
   * Automatically adjusts n to support efficient bootstrapping
   */
  static CKKSParams default_params_with_bootstrap(int mult_depth);
  
  /**
   * Validate parameters meet security requirements
   */
  static bool validate_security(const CKKSParams& params, SecurityLevel sec_level);
  
private:
  // Security standard: log2(n) -> max total log(Q) for given security
  static const std::vector<std::pair<int, int>> security_standard_128;
  static const std::vector<std::pair<int, int>> security_standard_192;
  static const std::vector<std::pair<int, int>> security_standard_256;
  
  static int get_max_log_q(int log_n, SecurityLevel sec_level);
};

// Output operators
std::ostream& operator<<(std::ostream& os, const CKKSParams& params);
std::ostream& operator<<(std::ostream& os, CKKSParamSelector::SecurityLevel level);

} // namespace fheco::ckks

