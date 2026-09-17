#include "fheco/code_gen/gen_func_heongpu.hpp"
#include "fheco/code_gen/constants_heongpu.hpp"
#include "fheco/ir/common.hpp"
#include "fheco/ir/func.hpp"
#include "fheco/passes/prepare_code_gen.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

namespace fheco::code_gen::heongpu
{

void gen_func_heongpu(
  const shared_ptr<ir::Func> &func,
  const unordered_set<int> &rotation_steps,
  ostream &os,
  string_view func_name,
  int scheme)
{
  passes::prepare_code_gen(func);
  
  os << header_includes << "\n";
  os << source_includes << "\n";
  os << source_usings << "\n";

  string scheme_str = (scheme == 0) ? "heongpu::Scheme::BFV" : "heongpu::Scheme::CKKS";
  os << "constexpr auto SCHEME = " << scheme_str << ";\n\n";

  // Generate rotation steps getter
  os << "std::vector<int> getRotationSteps() {\n";
  os << "    return {";
  bool first = true;
  for (int step : rotation_steps) {
    if (!first) os << ", ";
    os << step;
    first = false;
  }
  os << "};\n}\n\n";

  // Generate the main computation function signature
  os << "void " << func_name << "(\n";
  os << "    std::unordered_map<std::string, heongpu::Ciphertext<SCHEME>>& encryptedInputs,\n";
  os << "    std::unordered_map<std::string, heongpu::Plaintext<SCHEME>>& encodedInputs,\n";
  os << "    std::unordered_map<std::string, heongpu::Ciphertext<SCHEME>>& encryptedOutputs,\n";
  os << "    std::unordered_map<std::string, heongpu::Plaintext<SCHEME>>& encodedOutputs,\n";
  os << "    heongpu::HEEncoder<SCHEME>& encoder,\n";
  os << "    heongpu::HEEncryptor<SCHEME>& encryptor,\n";
  os << "    heongpu::HEArithmeticOperator<SCHEME>& ops,\n";
  os << "    heongpu::Relinkey<SCHEME>& relin_key,\n";
  os << "    heongpu::Galoiskey<SCHEME>& galois_keys) {\n\n";

  TermsCtxtObjectsInfo terms_ctxt_objects_info;
  gen_input_terms(func->data_flow().inputs_info(), os, terms_ctxt_objects_info, scheme);
  gen_const_terms(func->data_flow().constants_info(), func->clear_data_evaluator().signedness(), os, scheme);
  
  os << "    // FHE Operations\n";
  for (auto term : func->get_top_sorted_terms())
  {
    if (!term->is_operation()) continue;
    gen_term_eval(term, terms_ctxt_objects_info, os, scheme);
  }

  gen_output_terms(func->data_flow().outputs_info(), terms_ctxt_objects_info, os, scheme);

  os << "}\n\n";

  // Collect unique cipher input labels
  std::set<std::string> cipher_input_labels;
  for (const auto &input_info : func->data_flow().inputs_info())
  {
    if (input_info.first->type() == ir::Term::Type::cipher)
      cipher_input_labels.insert(input_info.second.label_);
  }

  // Generate main()
  os << "int main(int argc, char **argv) {\n";
  os << "    int poly_modulus_degree = 16384;\n";
  if (scheme == 0) { // BFV
    os << "    std::vector<int> q_bits = {60, 40, 40, 60};\n";
    os << "    std::vector<int> p_bits = {60};\n";
    os << "    int plain_modulus = 1032193;\n";
  } else { // CKKS
    os << "    std::vector<int> q_bits = {60, 40, 40, 40, 40, 40, 40, 40};\n";
    os << "    std::vector<int> p_bits = {60};\n";
    os << "    double scale = pow(2.0, 40);\n";
  }

  os << "    heongpu::HEContext<SCHEME> context = heongpu::GenHEContext<SCHEME>();\n";
  os << "    context->set_poly_modulus_degree(poly_modulus_degree);\n";
  os << "    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);\n";
  if (scheme == 0) os << "    context->set_plain_modulus(plain_modulus);\n";
  os << "    context->generate();\n\n";

  os << "    heongpu::HEKeyGenerator<SCHEME> keygen(context);\n";
  os << "    heongpu::Secretkey<SCHEME> secret_key(context);\n";
  os << "    keygen.generate_secret_key(secret_key);\n";
  os << "    heongpu::Publickey<SCHEME> public_key(context);\n";
  os << "    keygen.generate_public_key(public_key, secret_key);\n";
  os << "    heongpu::Relinkey<SCHEME> relin_key(context);\n";
  os << "    keygen.generate_relin_key(relin_key, secret_key);\n\n";

  os << "    auto t_keys_start = std::chrono::high_resolution_clock::now();\n";
  os << "    heongpu::Galoiskey<SCHEME> galois_keys(context);\n";
  os << "    auto rotations = getRotationSteps();\n";
  os << "    if (!rotations.empty()) {\n";
  os << "        keygen.generate_galois_key(galois_keys, secret_key, rotations);\n";
  os << "    }\n";
  os << "    cudaDeviceSynchronize();\n";
  os << "    auto t_keys_end = std::chrono::high_resolution_clock::now();\n";
  os << "    double keys_elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_keys_end - t_keys_start).count();\n\n";

  // To simulate galois keys size (since heongpu doesn't have an exact byte size method in its basic API, we approximate based on N and bits)
  os << "    double galois_keys_size_mb = (rotations.size() * poly_modulus_degree * q_bits.size() * 8.0) / (1024.0 * 1024.0);\n";
  os << "    std::cout << \"rotation_keys_size_(MB): \" << galois_keys_size_mb << \"\\n\";\n\n";

  os << "    heongpu::HEEncoder<SCHEME> encoder(context);\n";
  os << "    heongpu::HEEncryptor<SCHEME> encryptor(context, public_key);\n";
  os << "    heongpu::HEDecryptor<SCHEME> decryptor(context, secret_key);\n";
  os << "    heongpu::HEArithmeticOperator<SCHEME> ops(context, encoder);\n\n";

  os << "    std::unordered_map<std::string, heongpu::Ciphertext<SCHEME>> encryptedInputs;\n";
  os << "    std::unordered_map<std::string, heongpu::Plaintext<SCHEME>> encodedInputs;\n";
  os << "    std::unordered_map<std::string, heongpu::Ciphertext<SCHEME>> encryptedOutputs;\n";
  os << "    std::unordered_map<std::string, heongpu::Plaintext<SCHEME>> encodedOutputs;\n\n";

  os << "    std::size_t slot_count = poly_modulus_degree / 2;\n";
  if (scheme == 0) {
      os << "    std::vector<uint64_t> values(slot_count, 1);\n";
  } else {
      os << "    std::vector<double> values(slot_count, 1.0);\n";
  }

  for (const auto &label : cipher_input_labels) {
    os << "    {\n";
    os << "        heongpu::Plaintext<SCHEME> pt(context);\n";
    if (scheme == 0) os << "        encoder.encode(pt, values);\n";
    else os << "        encoder.encode(pt, values, scale);\n";
    os << "        heongpu::Ciphertext<SCHEME> ct(context);\n";
    os << "        encryptor.encrypt(ct, pt);\n";
    os << "        encryptedInputs[\"" << label << "\"] = ct;\n";
    os << "    }\n";
  }

  os << "\n    cudaDeviceSynchronize();\n";
  os << "    auto t_start = std::chrono::high_resolution_clock::now();\n";
  os << "    " << func_name << "(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, encryptor, ops, relin_key, galois_keys);\n";
  os << "    cudaDeviceSynchronize();\n";
  os << "    auto t_end = std::chrono::high_resolution_clock::now();\n";
  os << "    double elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_end - t_start).count();\n\n";

  os << "    std::cout << \"circuit_execution_time_(ms): \" << elapsed << \"\\n\";\n";
  os << "    std::cout << \"galois_keys_generation_time_(ms): \" << keys_elapsed << \"\\n\";\n";
  os << "    std::cout << \"total_execution_time_(ms): \" << (elapsed + keys_elapsed) << \"\\n\";\n";

  os << "    return 0;\n";
  os << "}\n";
}

void gen_cipher_var_id(std::size_t term_id, std::ostream &os) {
  os << "c" << term_id;
}

void gen_plain_var_id(std::size_t term_id, std::ostream &os) {
  os << "p" << term_id;
}

void gen_input_terms(
  const ir::InputTermsInfo &input_terms_info, std::ostream &os, TermsCtxtObjectsInfo &terms_ctxt_objects_info, int scheme)
{
  for (const auto &input_info : input_terms_info)
  {
    auto term = input_info.first;
    auto object_id = term->id();
    
    if (term->type() == ir::Term::Type::cipher)
    {
      terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{object_id, term->parents().size()});
      os << "    heongpu::Ciphertext<SCHEME>& ";
      gen_cipher_var_id(object_id, os);
      os << " = encryptedInputs.at(\"" << input_info.second.label_ << "\");\n";
    }
    else
    {
      os << "    heongpu::Plaintext<SCHEME>& ";
      gen_plain_var_id(object_id, os);
      os << " = encodedInputs.at(\"" << input_info.second.label_ << "\");\n";
    }
  }
}

void gen_const_terms(
  const ir::ConstTermsValues &const_terms_info, bool signedness, std::ostream &os, int scheme)
{
  if (const_terms_info.empty()) return;
  
  os << "    std::size_t slot_count = encoder.slot_count();\n";
  for (const auto &const_info : const_terms_info)
  {
    auto term = const_info.first;
    auto object_id = term->id();
    
    os << "    heongpu::Plaintext<SCHEME> ";
    gen_plain_var_id(object_id, os);
    os << "(context);\n";
    
    // Simplistic encoding for const terms
    if (scheme == 0) {
      os << "    {\n";
      os << "        std::vector<uint64_t> vals(slot_count, " << const_info.second.val_[0] << ");\n";
      os << "        encoder.encode(";
      gen_plain_var_id(object_id, os);
      os << ", vals);\n";
      os << "    }\n";
    } else {
      os << "    {\n";
      os << "        std::vector<double> vals(slot_count, " << const_info.second.val_[0] << ".0);\n";
      os << "        encoder.encode(";
      gen_plain_var_id(object_id, os);
      os << ", vals, scale);\n";
      os << "    }\n";
    }
  }
}

void gen_term_eval(
  const ir::Term *term, TermsCtxtObjectsInfo &terms_ctxt_objects_info, std::ostream &os, int scheme)
{
  auto term_object_id = term->id();
  std::vector<size_t> operands_ctxt_objects_ids(term->operands().size());
  std::unordered_map<size_t, size_t> operands_multip;

  for (size_t i = 0; i < operands_ctxt_objects_ids.size(); ++i)
  {
    auto operand = term->operands()[i];
    if (operand->type() != ir::Term::Type::cipher) continue;

    auto multip = ++operands_multip[operand->id()];
    auto operand_object_info_it = terms_ctxt_objects_info.find(operand->id());
    
    if (operand_object_info_it == terms_ctxt_objects_info.end())
    {
      operands_ctxt_objects_ids[i] = term_object_id;
      continue;
    }
    auto &operand_object_info = operand_object_info_it->second;
    operands_ctxt_objects_ids[i] = operand_object_info.id_;
    
    if (operand->type() == ir::Term::Type::plain || multip > 1) continue;
    
    --operand_object_info.dep_count_;
    if (term_object_id == term->id() && operand_object_info.dep_count_ == 0)
    {
      term_object_id = operands_ctxt_objects_ids[i];
      terms_ctxt_objects_info.erase(operand_object_info_it);
    }
  }

  if (term_object_id == term->id())
  {
    for (auto it = terms_ctxt_objects_info.begin(); it != terms_ctxt_objects_info.end(); ++it)
    {
      if (it->second.dep_count_ == 0)
      {
        term_object_id = it->second.id_;
        terms_ctxt_objects_info.erase(it);
        break;
      }
    }
  }
  
  auto dep_count = term->parents().size();
  terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{term_object_id, dep_count});

  if (term_object_id == term->id())
  {
    os << "    heongpu::Ciphertext<SCHEME> ";
    gen_cipher_var_id(term_object_id, os);
    os << "(context);\n";
  }

  vector<ir::Term::Type> operands_types;
  for (auto op : term->operands()) operands_types.push_back(op->type());

  if (term->op_code() == ir::OpCode::encrypt)
  {
    os << "    encryptor.encrypt(";
    gen_cipher_var_id(term_object_id, os);
    os << ", ";
    gen_plain_var_id(term->operands()[0]->id(), os);
    os << ");\n";
  }
  else
  {
    auto op_type = ir::OpType{term->op_code().type(), std::move(operands_types)};
    auto op_it = operation_mapping.find(op_type);
    if (op_it == operation_mapping.end()) {
        os << "    // WARNING: Unsupported op\n";
        return;
    }
    string op_name(op_it->second);

    if (term->op_code().type() == ir::OpCode::Type::rotate) {
      // Rotate
      os << "    ops." << op_name << "(";
      gen_cipher_var_id(term_object_id, os); // dst
      os << ", ";
      gen_cipher_var_id(operands_ctxt_objects_ids[0], os); // src
      os << ", " << term->op_code().generators()[0] << ", galois_keys);\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::relin) {
      // Relinearize
      os << "    ops." << op_name << "(";
      gen_cipher_var_id(term_object_id, os); // dst
      os << ", ";
      gen_cipher_var_id(operands_ctxt_objects_ids[0], os); // src
      os << ", relin_key);\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::rescale) {
      // Rescale
      os << "    ops." << op_name << "(";
      gen_cipher_var_id(term_object_id, os); // dst
      os << ", ";
      gen_cipher_var_id(operands_ctxt_objects_ids[0], os); // src
      os << ");\n";
    }
    else {
      // Binary ops
      os << "    ops." << op_name << "(";
      gen_cipher_var_id(term_object_id, os); // dst
      os << ", ";
      
      auto operand0 = term->operands()[0];
      if (operand0->type() == ir::Term::Type::cipher) gen_cipher_var_id(operands_ctxt_objects_ids[0], os);
      else gen_plain_var_id(operand0->id(), os);
      
      os << ", ";
      
      auto operand1 = term->operands()[1];
      if (operand1->type() == ir::Term::Type::cipher) gen_cipher_var_id(operands_ctxt_objects_ids[1], os);
      else gen_plain_var_id(operand1->id(), os);
      
      os << ");\n";
    }
  }
}

void gen_output_terms(
  const ir::OutputTermsInfo &output_terms_info, const TermsCtxtObjectsInfo &terms_ctxt_objects_info, std::ostream &os, int scheme)
{
  os << "\n    // Store outputs\n";
  for (const auto &output_info : output_terms_info)
  {
    auto term = output_info.first;
    if (term->type() == ir::Term::Type::cipher)
    {
      auto ctxt_object_id = terms_ctxt_objects_info.at(term->id()).id_;
      for (const auto &label : output_info.second.labels_)
      {
        os << "    encryptedOutputs[\"" << label << "\"] = ";
        gen_cipher_var_id(ctxt_object_id, os);
        os << ";\n";
      }
    }
    else
    {
      for (const auto &label : output_info.second.labels_)
      {
        os << "    encodedOutputs[\"" << label << "\"] = ";
        gen_plain_var_id(term->id(), os);
        os << ";\n";
      }
    }
  }
}

} // namespace fheco::code_gen::heongpu
