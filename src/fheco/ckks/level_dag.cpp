#include "fheco/ckks/level_dag.hpp"
#include "fheco/ir/op_code.hpp"
#include <algorithm>
#include <queue>
#include <iostream>
#include <cmath>

namespace fheco::ckks
{

// ==================== LevelDAG Implementation ====================

LevelDAG::LevelDAG(int l_eff, const std::shared_ptr<ir::Func>& func, const CKKSParams& params)
  : l_eff_(l_eff), func_(func), params_(params)
{
}

int LevelDAG::estimate_term_depth(const ir::Term* term) const
{
  // NOTE: This returns the LOCAL depth (how many levels THIS term consumes)
  // NOT the cumulative depth from inputs
  
  auto op_type = term->op_code().type();
  
  // Base case: input/leaf
  if (term->is_leaf() || op_type == ir::OpCode::Type::encrypt)
    return 0;
  
  // Operations that consume levels
  if (op_type == ir::OpCode::Type::mul)
  {
    auto arg1 = term->operands()[0];
    auto arg2 = term->operands()[1];
    
    // Only cipher-cipher multiplication consumes a level
    if (arg1->type() == ir::Term::Type::cipher && 
        arg2->type() == ir::Term::Type::cipher)
      return 1;
    return 0;
  }
  
  if (op_type == ir::OpCode::Type::square)
    return 1;
  
  // Other operations don't consume levels
  return 0;
}

int LevelDAG::get_cumulative_depth(const ir::Term* term) const
{
  // This returns the cumulative depth from inputs (cached)
  auto it = term_depths_.find(term->id());
  if (it != term_depths_.end())
    return it->second;
  return 0;
}

void LevelDAG::build()
{
  // Step 1: Compute cumulative depth for each term
  for (auto* term : func_->get_top_sorted_terms())
  {
    if (term->type() != ir::Term::Type::cipher)
      continue;
    
    // Compute cumulative depth from inputs
    int max_operand_depth = 0;
    for (auto* operand : term->operands())
    {
      if (operand->type() == ir::Term::Type::cipher)
      {
        auto it = term_depths_.find(operand->id());
        if (it != term_depths_.end())
          max_operand_depth = std::max(max_operand_depth, it->second);
      }
    }
    
    int term_local_depth = estimate_term_depth(term);
    int term_depth = max_operand_depth + term_local_depth;
    term_depths_[term->id()] = term_depth;
  }
  
  // Step 2: Build level DAG nodes
  // For each cipher term, create nodes at all possible levels
  auto sorted_terms = func_->get_top_sorted_terms();
  std::vector<const ir::Term*> cipher_terms;
  
  for (auto* term : sorted_terms)
  {
    if (term->type() != ir::Term::Type::cipher)
      continue;
    
    cipher_terms.push_back(term);
    
    for (int level = 0; level <= l_eff_; ++level)
    {
      LevelDAGNode node;
      node.term_id = term->id();
      node.level = level;
      node.weight = estimate_layer_latency(term, level);
      add_node(node);
    }
  }
  
  // Step 3: Build edges between consecutive terms in topological order
  // Key insight: we need to connect each term to ALL its cipher operands
  
  for (auto* term : cipher_terms)
  {
    if (term->is_leaf())
      continue;
    
    // Get cipher operands
    std::vector<const ir::Term*> cipher_operands;
    for (auto* operand : term->operands())
    {
      if (operand->type() == ir::Term::Type::cipher)
        cipher_operands.push_back(operand);
    }
    
    if (cipher_operands.empty())
      continue;
    
    // Get local depth consumed by this operation
    int local_depth = estimate_term_depth(term);
    
    // For each level assignment of this term (output level)
    for (int curr_level = 0; curr_level <= l_eff_; ++curr_level)
    {
      LevelDAGNode curr_node;
      curr_node.term_id = term->id();
      curr_node.level = curr_level;
      
      // Connect from each operand
      for (auto* operand : cipher_operands)
      {
        for (int prev_level = 0; prev_level <= l_eff_; ++prev_level)
        {
          LevelDAGNode prev_node;
          prev_node.term_id = operand->id();
          prev_node.level = prev_level;
          
          // Calculate edge weight based on Orion's logic:
          // - prev_level is the level of the operand (input to current op)
          // - curr_level is the desired output level of current op
          // - local_depth is levels consumed by current operation
          //
          // Valid transition WITHOUT bootstrap:
          //   curr_level == prev_level - local_depth
          //   (output level = input level - depth consumed)
          //
          // If curr_level > prev_level - local_depth:
          //   Need bootstrap to restore levels before this operation
          
          double cost = 0;
          int num_boots = 0;
          
          int output_level_without_boot = prev_level - local_depth;
          
          if (local_depth > 0)
          {
            // Level-consuming operation (mul, square)
            if (curr_level == output_level_without_boot && curr_level >= 0)
            {
              // Valid transition without bootstrap
              cost = 0;
            }
            else if (curr_level > output_level_without_boot)
            {
              // Need bootstrap: after bootstrap, level goes to l_eff
              // Then after operation, level becomes l_eff - local_depth
              // This is only valid if curr_level == l_eff - local_depth
              if (curr_level == l_eff_ - local_depth && prev_level > 0)
              {
                // Bootstrap + operation
                double a = 3.41, b = 0.18, c = 4.81;
                cost = a * std::exp(b * l_eff_) + c;
                num_boots = 1;
              }
              else
              {
                // Invalid transition
                cost = INF;
              }
            }
            else
            {
              // curr_level < output_level_without_boot: can use mod_switch
              // but we prefer exact level matching for simplicity
              cost = INF;  // Force exact level tracking
            }
          }
          else
          {
            // Non-level-consuming operation
            if (curr_level == prev_level)
            {
              // Level preserved
              cost = 0;
            }
            else if (curr_level == l_eff_ && prev_level > 0)
            {
              // Bootstrap to restore to max level
              double a = 3.41, b = 0.18, c = 4.81;
              cost = a * std::exp(b * l_eff_) + c;
              num_boots = 1;
            }
            else if (curr_level < prev_level)
            {
              // Can use mod_switch to lower level (free)
              cost = 0;
            }
            else
            {
              // Invalid: can't increase level without bootstrap
              cost = INF;
            }
          }
          
          add_edge(prev_node, curr_node, cost, num_boots);
        }
      }
    }
  }
}

double LevelDAG::estimate_layer_latency(const ir::Term* term, int level) const
{
  if (!term)
    return 0;
  
  auto it = term_depths_.find(term->id());
  int depth = (it != term_depths_.end()) ? it->second : 0;
  
  // If level < depth needed, this assignment is impossible
  if (level < depth)
    return INF;
  
  // Simple latency model: proportional to level (more levels = faster)
  // In practice, could use more sophisticated model like Orion
  return 0.001 * (l_eff_ - level);
}

std::pair<double, int> LevelDAG::estimate_bootstrap_cost(
  const ir::Term* prev_term, int prev_level,
  const ir::Term* curr_term, int curr_level) const
{
  if (!prev_term || !curr_term)
    return {0, 0};
  
  auto it = term_depths_.find(prev_term->id());
  int prev_depth = (it != term_depths_.end()) ? it->second : 0;
  
  // Level after previous term executes
  int level_after_prev = prev_level - estimate_term_depth(prev_term);
  
  // Check if we need bootstrap
  // Bootstrap needed if: curr_level > level_after_prev
  // i.e., current term needs more levels than what's available after previous
  
  if (curr_level > level_after_prev)
  {
    // Bootstrap required
    if (level_after_prev <= 0)
    {
      // Cannot bootstrap - would have negative levels
      return {INF, 0};
    }
    
    // Bootstrap cost formula from Orion: a * exp(b * l_eff) + c
    double a = 3.41, b = 0.18, c = 4.81;
    double t_boot = a * std::exp(b * l_eff_) + c;
    
    return {t_boot, 1};
  }
  
  // No bootstrap needed
  return {0, 0};
}

void LevelDAG::add_node(const LevelDAGNode& node)
{
  nodes_.insert(node);
  std::string key = node.to_string();
  if (adj_list_.find(key) == adj_list_.end())
  {
    adj_list_[key] = {};
    rev_adj_list_[key] = {};
  }
}

void LevelDAG::add_edge(const LevelDAGNode& from, const LevelDAGNode& to, 
                        double weight, int num_boots)
{
  edges_.push_back({from, to, weight, num_boots, {from, to}});
  adj_list_[from.to_string()].push_back(&edges_.back());
  rev_adj_list_[to.to_string()].push_back(&edges_.back());
}

std::vector<LevelDAGNode> LevelDAG::get_head_nodes() const
{
  std::vector<LevelDAGNode> heads;
  for (const auto& node : nodes_)
  {
    auto it = rev_adj_list_.find(node.to_string());
    if (it == rev_adj_list_.end() || it->second.empty())
    {
      heads.push_back(node);
    }
  }
  return heads;
}

std::vector<LevelDAGNode> LevelDAG::get_tail_nodes() const
{
  std::vector<LevelDAGNode> tails;
  for (const auto& node : nodes_)
  {
    auto it = adj_list_.find(node.to_string());
    if (it == adj_list_.end() || it->second.empty())
    {
      tails.push_back(node);
    }
  }
  return tails;
}

std::pair<std::vector<LevelDAGNode>, double> LevelDAG::shortest_path()
{
  // Dijkstra's algorithm with topological sort relaxation
  // Based on Orion's level_dag.py shortest_path implementation
  
  std::unordered_map<std::string, double> distances;
  std::unordered_map<std::string, std::string> predecessors;
  std::unordered_map<std::string, int> bootstrap_counts;
  
  // Initialize distances to infinity
  for (const auto& node : nodes_)
  {
    std::string key = node.to_string();
    distances[key] = INF;
    predecessors[key] = "";
    bootstrap_counts[key] = 0;
  }
  
  // Get topological order of terms
  auto sorted_terms = func_->get_top_sorted_terms();
  std::vector<const ir::Term*> cipher_terms;
  for (auto* term : sorted_terms)
  {
    if (term->type() == ir::Term::Type::cipher)
      cipher_terms.push_back(term);
  }
  
  if (cipher_terms.empty())
  {
    return {{}, 0};
  }
  
  // Initialize head nodes (input terms) with their weights
  auto* first_term = cipher_terms[0];
  for (int level = 0; level <= l_eff_; ++level)
  {
    LevelDAGNode head;
    head.term_id = first_term->id();
    head.level = level;
    std::string key = head.to_string();
    
    // Find node weight
    for (const auto& n : nodes_)
    {
      if (n.to_string() == key)
      {
        distances[key] = n.weight;
        predecessors[key] = "source";
        break;
      }
    }
  }
  
  // Also initialize any other input/leaf cipher terms
  for (auto* term : cipher_terms)
  {
    if (term->is_leaf())
    {
      for (int level = 0; level <= l_eff_; ++level)
      {
        LevelDAGNode head;
        head.term_id = term->id();
        head.level = level;
        std::string key = head.to_string();
        
        for (const auto& n : nodes_)
        {
          if (n.to_string() == key && distances[key] == INF)
          {
            distances[key] = n.weight;
            predecessors[key] = "source";
            break;
          }
        }
      }
    }
  }
  
  // Relax edges in topological order
  for (auto* term : cipher_terms)
  {
    for (int level = 0; level <= l_eff_; ++level)
    {
      LevelDAGNode node;
      node.term_id = term->id();
      node.level = level;
      std::string key = node.to_string();
      
      if (distances[key] == INF)
        continue;
      
      auto it = adj_list_.find(key);
      if (it == adj_list_.end())
        continue;
      
      for (auto* edge : it->second)
      {
        if (edge->weight == INF)
          continue;
          
        std::string neighbor_key = edge->to.to_string();
        double neighbor_weight = 0;
        
        // Find neighbor node weight
        for (const auto& n : nodes_)
        {
          if (n.to_string() == neighbor_key)
          {
            neighbor_weight = n.weight;
            break;
          }
        }
        
        if (neighbor_weight == INF)
          continue;
        
        double dist = distances[key] + edge->weight + neighbor_weight;
        
        if (dist < distances[neighbor_key])
        {
          distances[neighbor_key] = dist;
          predecessors[neighbor_key] = key;
          bootstrap_counts[neighbor_key] = bootstrap_counts[key] + edge->num_bootstraps;
        }
      }
    }
  }
  
  // Find best tail node (last term in topological order)
  auto* last_term = cipher_terms.back();
  double best_dist = INF;
  std::string best_tail = "";
  int best_level = 0;
  
  for (int level = 0; level <= l_eff_; ++level)
  {
    LevelDAGNode tail;
    tail.term_id = last_term->id();
    tail.level = level;
    std::string key = tail.to_string();
    
    if (distances[key] < best_dist)
    {
      best_dist = distances[key];
      best_tail = key;
      best_level = level;
    }
  }
  
  // Reconstruct path
  std::vector<LevelDAGNode> path;
  std::string current = best_tail;
  
  while (!current.empty() && current != "source")
  {
    // Parse node from string
    auto at_pos = current.find("@l=");
    if (at_pos != std::string::npos)
    {
      LevelDAGNode node;
      node.term_id = std::stoull(current.substr(1, at_pos - 1));
      node.level = std::stoi(current.substr(at_pos + 3));
      path.push_back(node);
    }
    current = predecessors[current];
  }
  
  std::reverse(path.begin(), path.end());
  
  // Store results
  optimal_path_ = path;
  total_bootstraps_ = bootstrap_counts.count(best_tail) ? bootstrap_counts[best_tail] : 0;
  
  // Extract level assignments for all terms (not just those in path)
  // Propagate levels through the graph
  optimal_levels_.clear();
  for (const auto& node : path)
  {
    optimal_levels_[node.term_id] = node.level;
  }
  
  // Also assign levels to terms not in the optimal path
  // (they inherit from their best predecessor)
  for (auto* term : cipher_terms)
  {
    if (optimal_levels_.find(term->id()) == optimal_levels_.end())
    {
      // Find best level for this term
      int best_term_level = l_eff_;
      double best_term_dist = INF;
      
      for (int level = 0; level <= l_eff_; ++level)
      {
        LevelDAGNode node;
        node.term_id = term->id();
        node.level = level;
        std::string key = node.to_string();
        
        if (distances[key] < best_term_dist)
        {
          best_term_dist = distances[key];
          best_term_level = level;
        }
      }
      
      optimal_levels_[term->id()] = best_term_level;
    }
  }
  
  return {path, best_dist};
}

std::unordered_map<size_t, int> LevelDAG::get_optimal_levels()
{
  if (optimal_levels_.empty())
  {
    shortest_path();
  }
  return optimal_levels_;
}

std::vector<size_t> LevelDAG::get_bootstrap_locations()
{
  if (optimal_levels_.empty())
  {
    shortest_path();
  }
  
  bootstrap_locations_.clear();
  total_bootstraps_ = 0;
  
  // =================================================================
  // Orion-style bootstrap placement algorithm:
  // - Simulate execution in topological order over ALL cipher terms
  // - Propagate levels through non-level-consuming ops (add, sub, etc.)
  // - Bootstrap when a level-consuming op would hit level 0 but there
  //   are more level-consuming operations remaining
  // - After bootstrap, level resets to l_eff (max level)
  // =================================================================
  
  auto sorted_terms = func_->get_top_sorted_terms();
  std::unordered_map<size_t, int> current_levels;
  
  std::vector<const ir::Term*> cipher_terms;
  int total_level_consuming = 0;
  
  for (auto* term : sorted_terms)
  {
    if (term->type() != ir::Term::Type::cipher)
      continue;
    cipher_terms.push_back(term);
    
    int local_depth = estimate_term_depth(term);
    if (local_depth > 0 && !term->is_leaf())
      ++total_level_consuming;
  }
  
#ifdef FHECO_LOGGING
  std::clog << "  Level-consuming terms: " << total_level_consuming << "\n";
  std::clog << "  Total cipher terms: " << cipher_terms.size() << "\n";
  std::clog << "  l_eff (max level): " << l_eff_ << "\n";
#endif
  
  // Initialize leaf cipher terms at max level
  for (auto* term : cipher_terms)
  {
    if (term->is_leaf())
    {
      current_levels[term->id()] = l_eff_;
    }
  }
  
  int remaining_consuming = total_level_consuming;
  
  // Process ALL cipher terms in topological order so that levels propagate
  // correctly through non-level-consuming operations (add_plain, etc.)
  for (auto* term : cipher_terms)
  {
    if (term->is_leaf())
      continue;
    
    int local_depth = estimate_term_depth(term);
    bool is_level_consuming = (local_depth > 0);
    
    if (is_level_consuming)
      --remaining_consuming;
    
    bool has_more_ops = (remaining_consuming > 0);
    
    // Get minimum operand level
    int min_operand_level = l_eff_;
    const ir::Term* lowest_operand = nullptr;
    
    for (auto* operand : term->operands())
    {
      if (operand->type() == ir::Term::Type::cipher)
      {
        auto it = current_levels.find(operand->id());
        if (it != current_levels.end() && it->second < min_operand_level)
        {
          min_operand_level = it->second;
          lowest_operand = operand;
        }
      }
    }
    
    // Level after this operation (without bootstrap)
    int output_level = min_operand_level - local_depth;
    
    // Bootstrap when a level-consuming op would reach level 0 (or below)
    // AND there are more level-consuming operations remaining
    if (is_level_consuming && output_level < 1 && has_more_ops && lowest_operand != nullptr)
    {
      bootstrap_locations_.push_back(lowest_operand->id());
      ++total_bootstraps_;
      
#ifdef FHECO_LOGGING
      int old_level = current_levels.count(lowest_operand->id()) 
                        ? current_levels[lowest_operand->id()] : -1;
#endif
      current_levels[lowest_operand->id()] = l_eff_;
      
      // Recalculate output level after bootstrap
      min_operand_level = l_eff_;
      output_level = l_eff_ - local_depth;
      
#ifdef FHECO_LOGGING
      std::clog << "  Bootstrap #" << total_bootstraps_ << " on $" 
                << lowest_operand->id() << " (level " << old_level << " -> " << l_eff_ << ")\n";
#endif
    }
    
    current_levels[term->id()] = std::max(0, output_level);
  }
  
  // Copy all levels to optimal_levels_
  for (auto* term : cipher_terms)
  {
    auto it = current_levels.find(term->id());
    if (it != current_levels.end())
    {
      optimal_levels_[term->id()] = it->second;
    }
    else
    {
      optimal_levels_[term->id()] = l_eff_;
    }
  }
  
  return bootstrap_locations_;
}

void LevelDAG::print(std::ostream& os) const
{
  os << "LevelDAG:\n";
  os << "  l_eff = " << l_eff_ << "\n";
  os << "  Nodes: " << nodes_.size() << "\n";
  os << "  Edges: " << edges_.size() << "\n";
  
  if (!optimal_path_.empty())
  {
    os << "  Optimal path:\n";
    for (const auto& node : optimal_path_)
    {
      os << "    " << node.to_string() << "\n";
    }
    os << "  Total bootstraps: " << total_bootstraps_ << "\n";
  }
}

// ==================== BootstrapSolver Implementation ====================

BootstrapSolver::BootstrapSolver(const std::shared_ptr<ir::Func>& func, 
                                   const CKKSParams& params)
  : func_(func), params_(params)
{
  int l_eff = static_cast<int>(params.max_level());
  level_dag_ = std::make_unique<LevelDAG>(l_eff, func, params);
}

BootstrapSolver::SolveResult BootstrapSolver::solve()
{
  // Build the level DAG
  level_dag_->build();
  
  // Find shortest path
  auto [path, cost] = level_dag_->shortest_path();
  
  // Get results
  result_.level_assignments = level_dag_->get_optimal_levels();
  result_.bootstrap_locations = level_dag_->get_bootstrap_locations();
  result_.num_bootstraps = level_dag_->get_total_bootstraps();
  
  // Input level is the level of the first node in the path
  if (!path.empty())
  {
    result_.input_level = path[0].level;
  }
  else
  {
    result_.input_level = static_cast<int>(params_.max_level());
  }
  
  return result_;
}

void BootstrapSolver::print_solution(std::ostream& os) const
{
  os << "BootstrapSolver Solution:\n";
  os << "  Input level: " << result_.input_level << "\n";
  os << "  Num bootstraps: " << result_.num_bootstraps << "\n";
  os << "  Bootstrap locations: ";
  for (size_t loc : result_.bootstrap_locations)
  {
    os << "$" << loc << " ";
  }
  os << "\n";
  
  os << "  Level assignments:\n";
  for (const auto& [term_id, level] : result_.level_assignments)
  {
    os << "    $" << term_id << " -> level " << level << "\n";
  }
}

} // namespace fheco::ckks

