#pragma once

#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/ckks/ckks_params.hpp"
#include "fheco/ckks/level_dag.hpp"
#include <memory>
#include <unordered_map>
#include <ostream>

namespace fheco::ckks
{

/**
 * Information about a ciphertext's scale and level
 * 
 * In CKKS:
 * - level: Starts at max_level, decreases after rescale/mod_switch
 * - log_scale: log2 of the scale factor (typically 40)
 * 
 * After operations:
 * - mul(c1, c2): scale = scale1 * scale2, level = min(level1, level2)
 * - mul(c, p):   scale depends on plaintext encoding
 * - add(c1, c2): scales and levels must match!
 * - rescale(c):  scale /= prime, level -= 1
 * - mod_switch(c): level -= 1 (scale unchanged)
 */
struct CiphertextInfo
{
  int level;       // Current level in modulus chain (decreases)
  int log_scale;   // log2(scale) - using integer for simplicity
  
  bool operator==(const CiphertextInfo& other) const {
    return level == other.level && log_scale == other.log_scale;
  }
  
  bool operator!=(const CiphertextInfo& other) const {
    return !(*this == other);
  }
};

/**
 * CKKS Scale and Level Manager
 * 
 * This class performs:
 * 1. Scale/level analysis on the IR
 * 2. Insertion of rescale operations after multiplications
 * 3. Insertion of mod_switch operations to align levels before additions
 * 4. Automatic bootstrap placement when levels are exhausted
 * 5. Validation that the circuit is valid for CKKS
 * 
 * Based on the algorithm from Orion (https://github.com/baahl-nyu/orion):
 * - Build a level DAG
 * - Find shortest path to minimize bootstrap count
 * - Place bootstraps at optimal locations
 * 
 * Usage:
 *   CKKSScaleManager manager(func, params);
 *   manager.analyze_and_transform();
 */
class CKKSScaleManager
{
public:
  CKKSScaleManager(std::shared_ptr<ir::Func> func, const CKKSParams& params);
  
  /**
   * Main entry point: analyze the IR and insert necessary operations
   * 
   * This method:
   * 1. Computes level/scale for each term
   * 2. Inserts rescale after multiplications (if not already present)
   * 3. Inserts mod_switch to align levels before additions/multiplications
   * 4. Inserts bootstrap operations when levels are exhausted
   * 
   * @return Number of operations inserted
   */
  size_t analyze_and_transform();
  
  /**
   * Get the computed info for a term (after analysis)
   */
  const CiphertextInfo* get_info(const ir::Term* term) const;
  
  /**
   * Print analysis results for debugging
   */
  void print_analysis(std::ostream& os) const;
  
  /**
   * Check if a term needs level alignment with another
   */
  static bool needs_level_alignment(const CiphertextInfo& a, const CiphertextInfo& b);
  
  /**
   * Get number of bootstrap operations inserted
   */
  size_t get_bootstrap_count() const { return bootstrap_count_; }
  
  /**
   * Check if bootstrapping is required for this circuit
   */
  bool requires_bootstrap() const { return bootstrap_count_ > 0; }
  
  /**
   * Get minimum level encountered in the circuit
   */
  int get_min_level() const { return min_level_reached_; }
  
  /**
   * Enable/disable bootstrap insertion (default: true)
   */
  void set_enable_bootstrap(bool enable) { enable_bootstrap_ = enable; }
  
private:
  // Analysis functions
  void compute_initial_info();
  CiphertextInfo compute_term_info(const ir::Term* term);
  
  // Transformation functions
  size_t insert_rescale_operations();
  size_t insert_level_alignment();
  size_t insert_bootstrap_operations();
  
  // Helper to insert mod_switch and update tracking
  ir::Term* insert_mod_switch_for_term(ir::Term* term, int target_level);
  
  // Helper to insert bootstrap and update tracking
  ir::Term* insert_bootstrap_for_term(ir::Term* term);
  
  // Check if term already has rescale as parent
  bool has_rescale_parent(const ir::Term* term) const;
  
  // Check if term already has bootstrap as parent
  bool has_bootstrap_parent(const ir::Term* term) const;
  
  // Check if operation is a multiplication that needs rescale
  bool needs_rescale(const ir::Term* term) const;
  
  // Check if a term needs bootstrap (level too low for next operation)
  bool needs_bootstrap(const ir::Term* term) const;
  
  std::shared_ptr<ir::Func> func_;
  CKKSParams params_;
  
  // Tracking info for each ciphertext term
  std::unordered_map<size_t, CiphertextInfo> term_info_;
  
  // Track which terms have been processed
  std::unordered_set<size_t> processed_terms_;
  
  // Bootstrap statistics
  size_t bootstrap_count_ = 0;
  int min_level_reached_ = 0;
  
  // Configuration
  bool enable_bootstrap_ = true;
  int min_level_before_bootstrap_ = 1;  // Bootstrap when level drops to this
};

// Output operator for CiphertextInfo
std::ostream& operator<<(std::ostream& os, const CiphertextInfo& info);

} // namespace fheco::ckks

