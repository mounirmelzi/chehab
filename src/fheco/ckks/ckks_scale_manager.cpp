#include "fheco/ckks/ckks_scale_manager.hpp"
#include "fheco/ckks/level_dag.hpp"
#include "fheco/ir/op_code.hpp"
#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace fheco::ckks
{

CKKSScaleManager::CKKSScaleManager(std::shared_ptr<ir::Func> func, const CKKSParams& params)
  : func_(std::move(func)), params_(params)
{
}

size_t CKKSScaleManager::analyze_and_transform()
{
  size_t total_inserted = 0;
  bootstrap_count_ = 0;
  
  // Step 1: Compute initial level/scale info for all terms
  compute_initial_info();
  
  // Step 2: Insert rescale operations after multiplications
  total_inserted += insert_rescale_operations();
  
  // Step 3: Recompute info after rescale insertion
  term_info_.clear();
  processed_terms_.clear();
  compute_initial_info();
  
  // Step 4: Insert mod_switch for level alignment
  total_inserted += insert_level_alignment();
  
  // Step 5: Recompute info after mod_switch insertion
  term_info_.clear();
  processed_terms_.clear();
  compute_initial_info();
  
  // Step 6: Insert bootstrap operations if levels are exhausted
  // This implements the algorithm from Orion paper (Section 5.2)
  // which uses a level DAG to find optimal bootstrap placement
  total_inserted += insert_bootstrap_operations();
  
  return total_inserted;
}

void CKKSScaleManager::compute_initial_info()
{
  // Process terms in topological order
  for (auto term : func_->get_top_sorted_terms())
  {
    if (term->type() != ir::Term::Type::cipher)
      continue;
    
    auto info = compute_term_info(term);
    term_info_[term->id()] = info;
  }
}

CiphertextInfo CKKSScaleManager::compute_term_info(const ir::Term* term)
{
  CiphertextInfo info;
  
  // Base case: input ciphertext
  if (term->is_leaf() || term->op_code().type() == ir::OpCode::Type::encrypt)
  {
    // Fresh ciphertext starts at max level with default scale
    info.level = static_cast<int>(params_.max_level());
    info.log_scale = params_.log_scale;
    return info;
  }
  
  auto op_type = term->op_code().type();
  
  // Get operand info (for cipher operands)
  std::vector<const CiphertextInfo*> operand_infos;
  for (auto operand : term->operands())
  {
    if (operand->type() == ir::Term::Type::cipher)
    {
      auto it = term_info_.find(operand->id());
      if (it != term_info_.end())
        operand_infos.push_back(&it->second);
    }
  }
  
  if (operand_infos.empty())
  {
    // Shouldn't happen for cipher operations
    info.level = static_cast<int>(params_.max_level());
    info.log_scale = params_.log_scale;
    return info;
  }
  
  // Compute based on operation type
  switch (op_type)
  {
    case ir::OpCode::Type::mul:
    {
      auto arg1 = term->operands()[0];
      auto arg2 = term->operands()[1];
      
      if (arg1->type() == ir::Term::Type::cipher && arg2->type() == ir::Term::Type::cipher)
      {
        // Cipher-cipher multiplication: followed by auto-rescale in code generator
        // So we simulate: mul doubles scale, then rescale normalizes
        // Net effect: level decreases by 1, scale stays at log_scale
        auto& info1 = term_info_[arg1->id()];
        auto& info2 = term_info_[arg2->id()];
        
        // Level decreases by 1 after the implicit rescale
        info.level = std::min(info1.level, info2.level) - 1;
        // Scale returns to normal after rescale
        info.log_scale = params_.log_scale;
      }
      else
      {
        // Cipher-plain multiplication: no auto-rescale needed
        // Level stays same, scale increases
        info.level = operand_infos[0]->level;
        info.log_scale = operand_infos[0]->log_scale + params_.log_scale;
      }
      break;
    }
    
    case ir::OpCode::Type::square:
    {
      // Square is cipher-cipher, so also has auto-rescale
      auto& operand_info = term_info_[term->operands()[0]->id()];
      info.level = operand_info.level - 1; // Consumes one level after rescale
      info.log_scale = params_.log_scale; // Returns to normal after rescale
      break;
    }
    
    case ir::OpCode::Type::rescale:
    {
      auto& operand_info = term_info_[term->operands()[0]->id()];
      info.level = operand_info.level - 1;
      info.log_scale = operand_info.log_scale - params_.log_scale;
      
      if (info.level < 0 && !enable_bootstrap_)
        throw std::runtime_error("Rescale would result in negative level - circuit too deep");
      break;
    }
    
    case ir::OpCode::Type::mod_switch:
    {
      auto& operand_info = term_info_[term->operands()[0]->id()];
      info.level = operand_info.level - 1;
      info.log_scale = operand_info.log_scale;
      
      if (info.level < 0 && !enable_bootstrap_)
        throw std::runtime_error("ModSwitch would result in negative level - circuit too deep");
      break;
    }
    
    case ir::OpCode::Type::bootstrap:
    {
      // Bootstrap refreshes the ciphertext to max level
      // Scale is normalized to default scale
      info.level = static_cast<int>(params_.max_level());
      info.log_scale = params_.log_scale;
      break;
    }
    
    case ir::OpCode::Type::add:
    case ir::OpCode::Type::sub:
    {
      auto arg1 = term->operands()[0];
      auto arg2 = term->operands()[1];
      
      if (arg1->type() == ir::Term::Type::cipher && arg2->type() == ir::Term::Type::cipher)
      {
        auto& info1 = term_info_[arg1->id()];
        auto& info2 = term_info_[arg2->id()];
        
        // For addition, levels and scales should match
        // Take the minimum level (will need alignment if different)
        info.level = std::min(info1.level, info2.level);
        info.log_scale = info1.log_scale; // Assume aligned
      }
      else
      {
        // Cipher-plain: use cipher's info
        info = *operand_infos[0];
      }
      break;
    }
    
    case ir::OpCode::Type::negate:
    case ir::OpCode::Type::rotate:
    case ir::OpCode::Type::relin:
    case ir::OpCode::Type::SumVec:
    {
      // These operations don't change level or scale
      info = *operand_infos[0];
      break;
    }
    
    default:
      // Unknown operation - preserve operand info
      info = *operand_infos[0];
      break;
  }
  
  return info;
}

bool CKKSScaleManager::needs_rescale(const ir::Term* term) const
{
  auto op_type = term->op_code().type();
  
  if (op_type == ir::OpCode::Type::mul)
  {
    auto arg1 = term->operands()[0];
    auto arg2 = term->operands()[1];
    // Only cipher-cipher multiplication needs rescale
    return arg1->type() == ir::Term::Type::cipher && 
           arg2->type() == ir::Term::Type::cipher;
  }
  
  if (op_type == ir::OpCode::Type::square)
  {
    return term->operands()[0]->type() == ir::Term::Type::cipher;
  }
  
  return false;
}

bool CKKSScaleManager::has_rescale_parent(const ir::Term* term) const
{
  for (auto parent : term->parents())
  {
    if (parent->op_code().type() == ir::OpCode::Type::rescale)
      return true;
  }
  return false;
}

size_t CKKSScaleManager::insert_rescale_operations()
{
  size_t count = 0;
  
  // Collect terms that need rescale first (to avoid modifying while iterating)
  std::vector<ir::Term*> terms_needing_rescale;
  
  // Get terms in topological order
  auto term_ids = func_->get_top_sorted_terms_ids();
  
  for (auto id : term_ids)
  {
    auto term = func_->data_flow().get_term(id);
    if (!term || term->type() != ir::Term::Type::cipher)
      continue;
    
    // Check if this term needs rescale and doesn't already have one
    if (needs_rescale(term) && !has_rescale_parent(term))
    {
      terms_needing_rescale.push_back(term);
    }
  }
  
  // Now insert rescale operations
  // Note: We just insert the rescale operations without modifying outputs
  // The code generator will handle the proper flow
  for (auto term : terms_needing_rescale)
  {
    // Insert rescale operation as a child of this term
    // This is sufficient - the rescale will be in the graph
    func_->insert_op_term(ir::OpCode::rescale, {term});
    ++count;
  }
  
  return count;
}

size_t CKKSScaleManager::insert_level_alignment()
{
  // Note: Level alignment (mod_switch insertion) is currently done at runtime
  // by Lattigo's evaluator, which automatically handles level mismatches.
  // 
  // For explicit level alignment in the IR, we would need a more sophisticated
  // graph rewriting approach to avoid creating cycles.
  //
  // For now, we rely on:
  // 1. Proper rescale placement after multiplications
  // 2. Lattigo's automatic level handling for additions
  //
  // In the future, this could be enhanced with:
  // - Proper SSA-style IR transformations
  // - Level-aware term creation
  
  size_t count = 0;
  
  // Track level mismatches for reporting
  auto term_ids = func_->get_top_sorted_terms_ids();
  
  for (auto id : term_ids)
  {
    auto term = func_->data_flow().get_term(id);
    if (!term || term->type() != ir::Term::Type::cipher)
      continue;
    
    auto op_type = term->op_code().type();
    
    // Check operations that require level alignment
    bool needs_alignment = (op_type == ir::OpCode::Type::add || 
                            op_type == ir::OpCode::Type::sub);
    
    if (!needs_alignment || term->operands().size() < 2)
      continue;
    
    auto arg1 = term->operands()[0];
    auto arg2 = term->operands()[1];
    
    // Only check cipher-cipher operations
    if (arg1->type() != ir::Term::Type::cipher || 
        arg2->type() != ir::Term::Type::cipher)
      continue;
    
    auto it1 = term_info_.find(arg1->id());
    auto it2 = term_info_.find(arg2->id());
    
    if (it1 == term_info_.end() || it2 == term_info_.end())
      continue;
    
    // Just track the mismatch - Lattigo will handle at runtime
    if (it1->second.level != it2->second.level)
    {
      // In a production system, we might emit a warning here
      // For now, Lattigo handles this automatically
      ++count;
    }
  }
  
  return count; // Return count of mismatches detected (not ops inserted)
}

ir::Term* CKKSScaleManager::insert_mod_switch_for_term(ir::Term* term, int target_level)
{
  auto it = term_info_.find(term->id());
  if (it == term_info_.end())
    return term;
  
  int current_level = it->second.level;
  if (current_level <= target_level)
    return term;
  
  ir::Term* current = term;
  for (int level = current_level; level > target_level; --level)
  {
    auto mod_switch_term = func_->insert_op_term(ir::OpCode::mod_switch, {current});
    
    // Track the new term
    CiphertextInfo new_info;
    new_info.level = level - 1;
    new_info.log_scale = term_info_[current->id()].log_scale;
    term_info_[mod_switch_term->id()] = new_info;
    
    current = mod_switch_term;
  }
  
  return current;
}

const CiphertextInfo* CKKSScaleManager::get_info(const ir::Term* term) const
{
  auto it = term_info_.find(term->id());
  if (it != term_info_.end())
    return &it->second;
  return nullptr;
}

void CKKSScaleManager::print_analysis(std::ostream& os) const
{
  os << "CKKS Scale/Level Analysis:\n";
  os << "===========================\n";
  os << "Parameters: max_level=" << params_.max_level() 
     << ", log_scale=" << params_.log_scale << "\n";
  os << "Bootstrap count: " << bootstrap_count_ << "\n";
  os << "Min level reached: " << min_level_reached_ << "\n";
  os << "Bootstrap enabled: " << (enable_bootstrap_ ? "yes" : "no") << "\n\n";
  
  for (const auto& entry : term_info_)
  {
    os << "Term $" << entry.first << ": " << entry.second << "\n";
  }
}

bool CKKSScaleManager::needs_level_alignment(const CiphertextInfo& a, const CiphertextInfo& b)
{
  return a.level != b.level;
}

std::ostream& operator<<(std::ostream& os, const CiphertextInfo& info)
{
  os << "(level=" << info.level << ", log_scale=" << info.log_scale << ")";
  return os;
}

bool CKKSScaleManager::has_bootstrap_parent(const ir::Term* term) const
{
  for (auto parent : term->parents())
  {
    if (parent->op_code().type() == ir::OpCode::Type::bootstrap)
      return true;
  }
  return false;
}

bool CKKSScaleManager::needs_bootstrap(const ir::Term* term) const
{
  if (!enable_bootstrap_)
    return false;
  
  auto it = term_info_.find(term->id());
  if (it == term_info_.end())
    return false;
  
  // Check if level is at or below minimum threshold
  // We need at least 1 level to continue operations
  return it->second.level <= min_level_before_bootstrap_;
}

ir::Term* CKKSScaleManager::insert_bootstrap_for_term(ir::Term* term)
{
  // Create bootstrap operation
  auto bootstrap_term = func_->insert_op_term(ir::OpCode::bootstrap, {term});
  
  // Update info for the bootstrapped term
  CiphertextInfo new_info;
  new_info.level = static_cast<int>(params_.max_level());
  new_info.log_scale = params_.log_scale;
  term_info_[bootstrap_term->id()] = new_info;
  
  return bootstrap_term;
}

size_t CKKSScaleManager::insert_bootstrap_operations()
{
  if (!enable_bootstrap_)
    return 0;
  
  size_t count = 0;
  bootstrap_count_ = 0;
  
  // ==================== Orion-style Level DAG Algorithm ====================
  // Based on Section 5.2 of Orion paper: https://arxiv.org/pdf/2311.03470
  // 
  // This algorithm:
  // 1. Builds a Level DAG where each node is (term, level) pair
  // 2. Edge weights represent bootstrap cost (0 if no bootstrap needed)
  // 3. Finds shortest path to minimize total bootstrap cost
  // 4. Places bootstraps at optimal locations
  
#ifdef FHECO_LOGGING
  std::clog << "\n==> Running Orion-style Level DAG bootstrap solver...\n";
#endif
  
  // Create and run the bootstrap solver
  BootstrapSolver solver(func_, params_);
  auto result = solver.solve();
  
#ifdef FHECO_LOGGING
  std::clog << "  Input level: " << result.input_level << "\n";
  std::clog << "  Optimal bootstraps needed: " << result.num_bootstraps << "\n";
  std::clog << "  Bootstrap locations: ";
  for (size_t loc : result.bootstrap_locations)
  {
    std::clog << "$" << loc << " ";
  }
  std::clog << "\n";
#endif
  
  // Now insert bootstrap operations at the optimal locations
  for (size_t term_id : result.bootstrap_locations)
  {
    auto* term = func_->data_flow().get_term(term_id);
    if (!term || has_bootstrap_parent(term))
      continue;
    
    // Insert bootstrap operation
    auto bootstrap_term = func_->insert_op_term(ir::OpCode::bootstrap, {term});
    
    // Update info for the bootstrapped term
    CiphertextInfo new_info;
    new_info.level = static_cast<int>(params_.max_level());
    new_info.log_scale = params_.log_scale;
    term_info_[bootstrap_term->id()] = new_info;
    
    // Replace uses of term with bootstrapped term
    func_->replace_term_with(term, bootstrap_term);
    
    ++count;
    ++bootstrap_count_;
  }
  
  // Update level assignments based on solver result
  for (const auto& [term_id, level] : result.level_assignments)
  {
    auto it = term_info_.find(term_id);
    if (it != term_info_.end())
    {
      it->second.level = level;
    }
  }
  
  // Find the minimum level in the circuit
  min_level_reached_ = result.input_level;
  for (const auto& [term_id, level] : result.level_assignments)
  {
    min_level_reached_ = std::min(min_level_reached_, level);
  }
  
  return count;
}

} // namespace fheco::ckks

