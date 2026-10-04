#pragma once

#include "fheco/ckks/ckks_params.hpp"
#include "fheco/ir/common.hpp"
#include "fheco/param_select/enc_params.hpp"
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fheco::ir
{
class Func;
} // namespace fheco::ir

namespace fheco::code_gen::lattigo
{

struct CtxtObjectInfo
{
  std::size_t id_;
  std::size_t dep_count_;
};

using TermsCtxtObjectsInfo = std::unordered_map<std::size_t, CtxtObjectInfo>;

void gen_func_lattigo(
  const std::shared_ptr<ir::Func> &func, 
  const std::unordered_set<int> &rotation_steps, 
  std::ostream &os,
  const std::string &func_name,
  const ckks::CKKSParams* ckks_params = nullptr);

void gen_func_signature_go(const std::string &func_name, std::ostream &os);

void gen_input_terms_go(
  const ir::InputTermsInfo &input_terms_info, 
  std::ostream &os, 
  TermsCtxtObjectsInfo &terms_ctxt_objects_info);

void gen_const_terms_go(
  const ir::ConstTermsValues &const_terms_info, 
  bool signedness, 
  std::ostream &os);

void gen_op_terms_go(
  const std::shared_ptr<ir::Func> &func, 
  std::ostream &os, 
  TermsCtxtObjectsInfo &terms_ctxt_objects_info);

void gen_output_terms_go(
  const ir::OutputTermsInfo &output_terms_info, 
  std::ostream &os, 
  const TermsCtxtObjectsInfo &terms_ctxt_objects_info);

void gen_rotation_steps_getter_go(
  const std::string &func_name, 
  const std::unordered_set<int> &steps, 
  std::ostream &os);

void gen_main_go(
  const std::string &func_name,
  const std::unordered_set<int> &rotation_steps,
  std::ostream &os,
  const ckks::CKKSParams* ckks_params = nullptr);

void gen_cipher_var_id_go(std::size_t term_id, std::ostream &os);

void gen_plain_var_id_go(std::size_t term_id, std::ostream &os);

} // namespace fheco::code_gen::lattigo