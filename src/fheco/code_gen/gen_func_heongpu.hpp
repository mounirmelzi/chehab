#pragma once

#include "fheco/ir/common.hpp"
#include <cstddef>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fheco::ir
{
class Func;
} // namespace fheco::ir

namespace fheco::code_gen::heongpu
{

struct CtxtObjectInfo
{
  std::size_t id_;
  std::size_t dep_count_;
};

using TermsCtxtObjectsInfo = std::unordered_map<std::size_t, CtxtObjectInfo>;

void gen_func_heongpu(
  const std::shared_ptr<ir::Func> &func, const std::unordered_set<int> &rotation_steps,
  std::ostream &cu_os, std::string_view func_name, int scheme);

void gen_input_terms(
  const ir::InputTermsInfo &input_terms_info, std::ostream &os, TermsCtxtObjectsInfo &terms_ctxt_objects_info, int scheme);

void gen_cipher_var_id(std::size_t term_id, std::ostream &os);

void gen_plain_var_id(std::size_t term_id, std::ostream &os);

void gen_const_terms(const ir::ConstTermsValues &const_terms_info, bool signedness, std::ostream &os, int scheme);

void gen_op_term(
  const ir::Term *term, const std::vector<std::size_t> &operands_ids, TermsCtxtObjectsInfo &terms_ctxt_objects_info,
  std::ostream &os, int scheme);

void gen_sum_vec(
  const ir::Term *term, const std::vector<std::size_t> &operands_ids, TermsCtxtObjectsInfo &terms_ctxt_objects_info,
  std::ostream &os, int scheme);

void gen_term_eval(
  const ir::Term *term, TermsCtxtObjectsInfo &terms_ctxt_objects_info, std::ostream &os, int scheme);

void gen_output_terms(
  const ir::OutputTermsInfo &output_terms_info, const TermsCtxtObjectsInfo &terms_ctxt_objects_info, std::ostream &os, int scheme);

void gen_return_stat(std::ostream &os);

} // namespace fheco::code_gen::heongpu
