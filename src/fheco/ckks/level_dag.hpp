#pragma once

#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/ckks/ckks_params.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <vector>
#include <string>
#include <limits>
#include <ostream>

namespace fheco::ckks
{

/**
 * LevelDAG - Implementation of the Level Digraph from Orion paper (Section 5.2)
 * 
 * This creates a DAG where:
 * - Each node represents (term, level) pair: "term_id@l=level"
 * - Edge weights represent bootstrap cost (0 if no bootstrap, inf if impossible)
 * - Shortest path gives optimal level assignment minimizing total bootstrap cost
 * 
 * Based on: https://arxiv.org/pdf/2311.03470
 */

struct LevelDAGNode
{
  size_t term_id;
  int level;
  double weight;  // Layer latency estimate
  
  std::string to_string() const {
    return "t" + std::to_string(term_id) + "@l=" + std::to_string(level);
  }
  
  bool operator==(const LevelDAGNode& other) const {
    return term_id == other.term_id && level == other.level;
  }
};

struct LevelDAGNodeHash {
  size_t operator()(const LevelDAGNode& node) const {
    return std::hash<size_t>()(node.term_id) ^ (std::hash<int>()(node.level) << 1);
  }
};

struct LevelDAGEdge
{
  LevelDAGNode from;
  LevelDAGNode to;
  double weight;  // Bootstrap cost (0 = no bootstrap, inf = impossible)
  int num_bootstraps;
  std::vector<LevelDAGNode> path;  // For tracking through aggregated DAGs
};

class LevelDAG
{
public:
  static constexpr double INF = std::numeric_limits<double>::infinity();
  static constexpr double BOOTSTRAP_BASE_COST = 10.0;  // Base cost for bootstrap
  
  LevelDAG(int l_eff, const std::shared_ptr<ir::Func>& func, const CKKSParams& params);
  
  /**
   * Build level DAG from the IR function
   */
  void build();
  
  /**
   * Find shortest path (minimum bootstrap cost)
   * Returns: (path, total_cost)
   */
  std::pair<std::vector<LevelDAGNode>, double> shortest_path();
  
  /**
   * Get optimal level assignment for each term
   * Returns: map of term_id -> optimal_level
   */
  std::unordered_map<size_t, int> get_optimal_levels();
  
  /**
   * Get bootstrap locations (terms that need bootstrap after them)
   */
  std::vector<size_t> get_bootstrap_locations();
  
  /**
   * Get total number of bootstraps required
   */
  int get_total_bootstraps() const { return total_bootstraps_; }
  
  /**
   * Print the level DAG for debugging
   */
  void print(std::ostream& os) const;
  
  /**
   * Estimate multiplicative depth of a term
   */
  int estimate_term_depth(const ir::Term* term) const;
  int get_cumulative_depth(const ir::Term* term) const;
  
private:
  void add_node(const LevelDAGNode& node);
  void add_edge(const LevelDAGNode& from, const LevelDAGNode& to, double weight, int num_boots = 0);
  
  /**
   * Estimate bootstrap cost between two level assignments
   */
  std::pair<double, int> estimate_bootstrap_cost(
    const ir::Term* prev_term, int prev_level,
    const ir::Term* curr_term, int curr_level) const;
  
  /**
   * Estimate layer latency at a given level
   */
  double estimate_layer_latency(const ir::Term* term, int level) const;
  
  /**
   * Get head nodes (no incoming edges)
   */
  std::vector<LevelDAGNode> get_head_nodes() const;
  
  /**
   * Get tail nodes (no outgoing edges)
   */
  std::vector<LevelDAGNode> get_tail_nodes() const;
  
  int l_eff_;  // Effective number of levels
  std::shared_ptr<ir::Func> func_;
  CKKSParams params_;
  
  // Graph representation
  std::unordered_set<LevelDAGNode, LevelDAGNodeHash> nodes_;
  std::deque<LevelDAGEdge> edges_;
  
  // Adjacency lists
  std::unordered_map<std::string, std::vector<LevelDAGEdge*>> adj_list_;  // outgoing edges
  std::unordered_map<std::string, std::vector<LevelDAGEdge*>> rev_adj_list_;  // incoming edges
  
  // Results
  std::vector<LevelDAGNode> optimal_path_;
  std::unordered_map<size_t, int> optimal_levels_;
  std::vector<size_t> bootstrap_locations_;
  int total_bootstraps_ = 0;
  
  // Term info cache
  std::unordered_map<size_t, int> term_depths_;  // term_id -> multiplicative depth
};

/**
 * BootstrapSolver - Orchestrates the level DAG construction and solving
 * 
 * Based on Orion's auto_bootstrap.py
 */
class BootstrapSolver
{
public:
  BootstrapSolver(const std::shared_ptr<ir::Func>& func, const CKKSParams& params);
  
  /**
   * Solve for optimal bootstrap placement
   * Returns: (input_level, num_bootstraps, bootstrap_term_ids)
   */
  struct SolveResult {
    int input_level;
    int num_bootstraps;
    std::vector<size_t> bootstrap_locations;
    std::unordered_map<size_t, int> level_assignments;
  };
  
  SolveResult solve();
  
  /**
   * Get the level DAG for visualization
   */
  const LevelDAG& get_level_dag() const { return *level_dag_; }
  
  /**
   * Print solution for debugging
   */
  void print_solution(std::ostream& os) const;
  
private:
  std::shared_ptr<ir::Func> func_;
  CKKSParams params_;
  std::unique_ptr<LevelDAG> level_dag_;
  SolveResult result_;
};

} // namespace fheco::ckks

