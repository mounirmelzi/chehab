#include "fheco/code_gen/gen_func_lattigo.hpp"
#include "fheco/code_gen/constants_lattigo.hpp"
#include "fheco/ckks/ckks_params.hpp"
#include "fheco/ir/common.hpp"
#include "fheco/ir/func.hpp"
#include "fheco/passes/prepare_code_gen.hpp"
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>

using namespace std;

namespace fheco::code_gen::lattigo
{

void gen_func_lattigo(
  const shared_ptr<ir::Func> &func,
  const unordered_set<int> &rotation_steps,
  ostream &os,
  const string &func_name,
  const ckks::CKKSParams* ckks_params)
{
  passes::prepare_code_gen(func);
  
  // Write Go file header with imports (include bootstrap imports if needed)
  if (ckks_params && ckks_params->enable_bootstrap)
  {
    os << go_file_header_bootstrap;
  }
  else
  {
    os << go_file_header;
  }
  
  // Generate rotation steps getter
  gen_rotation_steps_getter_go(func_name, rotation_steps, os);
  os << "\n";
  
  // Generate the main computation function
  gen_func_signature_go(func_name, os);
  os << " {\n";
  
  TermsCtxtObjectsInfo terms_ctxt_objects_info;
  gen_input_terms_go(func->data_flow().inputs_info(), os, terms_ctxt_objects_info);
  gen_const_terms_go(func->data_flow().constants_info(), func->clear_data_evaluator().signedness(), os);
  gen_op_terms_go(func, os, terms_ctxt_objects_info);
  gen_output_terms_go(func->data_flow().outputs_info(), os, terms_ctxt_objects_info);
  
  os << "}\n\n";
  
  // Generate main function with setup (pass CKKS params)
  gen_main_go(func_name, rotation_steps, os, ckks_params);
}

void gen_func_signature_go(const string &func_name, ostream &os)
{
  os << "func " << func_name << "(\n";
  os << "\tencryptedInputs map[string]*rlwe.Ciphertext,\n";
  os << "\tencodedInputs map[string]*rlwe.Plaintext,\n";
  os << "\tencryptedOutputs map[string]*rlwe.Ciphertext,\n";
  os << "\tencodedOutputs map[string]*rlwe.Plaintext,\n";
  os << "\tencoder *hefloat.Encoder,\n";
  os << "\tenc *rlwe.Encryptor,\n";
  os << "\teval *hefloat.Evaluator,\n";
  os << "\tparams hefloat.Parameters,\n";
  os << ")";
}

void gen_cipher_var_id_go(size_t term_id, ostream &os)
{
  os << "c" << term_id;
}

void gen_plain_var_id_go(size_t term_id, ostream &os)
{
  os << "p" << term_id;
}

void gen_input_terms_go(
  const ir::InputTermsInfo &input_terms_info,
  ostream &os,
  TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  for (const auto &input_info : input_terms_info)
  {
    auto term = input_info.first;
    auto object_id = term->id();
    
    if (term->type() == ir::Term::Type::cipher)
    {
      terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{object_id, term->parents().size()});
      os << "\t";
      gen_cipher_var_id_go(object_id, os);
      os << " := encryptedInputs[\"" << input_info.second.label_ << "\"]\n";
    }
    else
    {
      os << "\t";
      gen_plain_var_id_go(object_id, os);
      os << " := encodedInputs[\"" << input_info.second.label_ << "\"]\n";
    }
  }
}

void gen_const_terms_go(
  const ir::ConstTermsValues &const_terms_info,
  bool signedness,
  ostream &os)
{
  if (const_terms_info.empty())
    return;
    
  os << "\n\t// Encode constants\n";
  os << "\tslotCount := params.MaxSlots()\n";
  
  for (const auto &const_info : const_terms_info)
  {
    auto term = const_info.first;
    auto object_id = term->id();
    
    os << "\t";
    gen_plain_var_id_go(object_id, os);
    os << " := hefloat.NewPlaintext(params, params.MaxLevel())\n";
    
    if (const_info.second.is_scalar_)
    {
      // Scalar constant - replicate across all slots
      os << "\t{\n";
      os << "\t\tvalues := make([]float64, slotCount)\n";
      os << "\t\tfor i := range values {\n";
      os << "\t\t\tvalues[i] = float64(" << const_info.second.val_[0] << ")\n";
      os << "\t\t}\n";
      os << "\t\tencoder.Encode(values, ";
      gen_plain_var_id_go(object_id, os);
      os << ")\n";
      os << "\t}\n";
    }
    else
    {
      // Vector constant
      os << "\t{\n";
      os << "\t\tvalues := []float64{";
      for (size_t i = 0; i < const_info.second.val_.size(); ++i)
      {
        if (i > 0) os << ", ";
        os << "float64(" << const_info.second.val_[i] << ")";
      }
      os << "}\n";
      os << "\t\tencoder.Encode(values, ";
      gen_plain_var_id_go(object_id, os);
      os << ")\n";
      os << "\t}\n";
    }
  }
}

void gen_op_terms_go(
  const shared_ptr<ir::Func> &func,
  ostream &os,
  TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  os << "\n\t// FHE Operations\n";
  
  for (auto term : func->get_top_sorted_terms())
  {
    if (!term->is_operation())
      continue;

    auto term_object_id = term->id();
    vector<size_t> operands_ctxt_objects_ids(term->operands().size());
    unordered_map<size_t, size_t> operands_multip;
    
    for (size_t i = 0; i < operands_ctxt_objects_ids.size(); ++i)
    {
      auto operand = term->operands()[i];
      if (operand->type() != ir::Term::Type::cipher)
        continue;

      auto multip = ++operands_multip[operand->id()];
      auto operand_object_info_it = terms_ctxt_objects_info.find(operand->id());
      
      if (operand_object_info_it == terms_ctxt_objects_info.end())
      {
        operands_ctxt_objects_ids[i] = term_object_id;
        continue;
      }
      
      auto &operand_object_info = operand_object_info_it->second;
      operands_ctxt_objects_ids[i] = operand_object_info.id_;
      
      if (func->data_flow().is_output(operand) || operand->type() == ir::Term::Type::plain || multip > 1)
        continue;

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
    if (func->data_flow().is_output(term))
      ++dep_count;

    terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{term_object_id, dep_count});

    // Declare new ciphertext if needed
    if (term_object_id == term->id())
    {
      os << "\tvar ";
      gen_cipher_var_id_go(term_object_id, os);
      os << " *rlwe.Ciphertext\n";
    }

    // Generate the operation
    vector<ir::Term::Type> operands_types;
    operands_types.reserve(term->operands().size());
    transform(
      term->operands().cbegin(), term->operands().cend(), back_inserter(operands_types),
      [](const ir::Term *operand) { return operand->type(); });

    if (term->op_code() == ir::OpCode::encrypt)
    {
      // Encrypt operation
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = enc.EncryptNew(";
      gen_plain_var_id_go(term->operands()[0]->id(), os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::rotate)
    {
      // Rotation: eval.Rotate(ct, k, ctOut)
      int steps = term->op_code().steps();
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval.RotateNew(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ", " << steps << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::square)
    {
      // Square: eval.MulRelinNew(ct, ct) + auto-rescale for CKKS
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval.MulRelinNew(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ", ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
      // Auto-rescale after square (cipher-cipher multiplication)
      os << "\t_ = eval.Rescale(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::rescale)
    {
      // Rescale: copy input, then rescale in-place
      // Lattigo's Rescale modifies in-place, so we need to copy first
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      os << "\t_ = eval.Rescale(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::relin)
    {
      // Relinearize: copy input, then relinearize in-place
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      os << "\t_ = eval.Relinearize(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::mod_switch)
    {
      // DropLevel
      os << "\teval.DropLevel(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ", 1)\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << "\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::SumVec)
    {
      // SumVec reduction: sum all slots using log(n) rotations and additions
      // SumVec(x, size) → x + (x << size/2) + ((x + (x << size/2)) << size/4) + ...
      int size = term->op_code().size();
      os << "\t// SumVec reduction (size=" << size << ")\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      
      // Generate log2(size) rotations and additions
      int step = size / 2;
      while (step >= 1)
      {
        os << "\t{\n";
        os << "\t\trotated, _ := eval.RotateNew(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", " << step << ")\n";
        os << "\t\t";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", _ = eval.AddNew(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", rotated)\n";
        os << "\t}\n";
        step /= 2;
      }
    }
    else if (term->op_code().type() == ir::OpCode::Type::bootstrap)
    {
      // Bootstrap: refresh ciphertext to max level
      // Requires bootstrapper to be initialized (see gen_main_go)
      os << "\t// Bootstrap: refresh to max level\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = bootstrapper.Bootstrap(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::negate)
    {
      // Negate: eval.NegNew(ct)
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval.NegNew(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
    }
    else
    {
      // Binary operations: add, sub, mul
      auto op_type = ir::OpType{term->op_code().type(), std::move(operands_types)};
      auto op_it = operation_mapping.find(op_type);
      
      if (op_it == operation_mapping.end())
      {
        os << "\t// WARNING: Unsupported operation " << term->op_code() << "\n";
        continue;
      }
      
      string op_name(op_it->second);
      
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval." << op_name << "New(";
      
      // First operand (always cipher for these ops)
      auto operand0 = term->operands()[0];
      if (operand0->type() == ir::Term::Type::cipher)
        gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      else
        gen_plain_var_id_go(operand0->id(), os);
      
      os << ", ";
      
      // Second operand
      auto operand1 = term->operands()[1];
      if (operand1->type() == ir::Term::Type::cipher)
        gen_cipher_var_id_go(operands_ctxt_objects_ids[1], os);
      else
        gen_plain_var_id_go(operand1->id(), os);
      
      os << ")\n";
      
      // Auto-rescale after cipher-cipher multiplication (CKKS)
      // MulRelin operations need rescale to maintain scale
      if (op_name == "MulRelin")
      {
        os << "\t_ = eval.Rescale(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", ";
        gen_cipher_var_id_go(term_object_id, os);
        os << ")\n";
      }
    }
  }
}

void gen_output_terms_go(
  const ir::OutputTermsInfo &output_terms_info,
  ostream &os,
  const TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  os << "\n\t// Store outputs\n";
  
  for (const auto &output_info : output_terms_info)
  {
    auto term = output_info.first;
    
    if (term->type() == ir::Term::Type::cipher)
    {
      auto ctxt_object_id = terms_ctxt_objects_info.at(term->id()).id_;
      
      for (const auto &label : output_info.second.labels_)
      {
        os << "\tencryptedOutputs[\"" << label << "\"] = ";
        gen_cipher_var_id_go(ctxt_object_id, os);
        os << "\n";
      }
    }
    else
    {
      for (const auto &label : output_info.second.labels_)
      {
        os << "\tencodedOutputs[\"" << label << "\"] = ";
        gen_plain_var_id_go(term->id(), os);
        os << "\n";
      }
    }
  }
}

void gen_rotation_steps_getter_go(
  const string &func_name,
  const unordered_set<int> &steps,
  ostream &os)
{
  os << "func getRotationSteps() []int {\n";
  os << "\treturn []int{";
  
  bool first = true;
  for (int step : steps)
  {
    if (!first) os << ", ";
    os << step;
    first = false;
  }
  
  os << "}\n";
  os << "}\n";
}

void gen_main_go(
  const string &func_name,
  const unordered_set<int> &rotation_steps,
  ostream &os,
  const ckks::CKKSParams* ckks_params)
{
  // Use provided params or create defaults
  ckks::CKKSParams params;
  if (ckks_params) {
    params = *ckks_params;
  } else {
    // Default params for depth ~7
    params = ckks::CKKSParamSelector::default_params(7);
  }
  
  os << "\nfunc main() {\n";
  os << "\t// CKKS Parameters (generated from CKKSParamSelector)\n";
  os << "\t// LogN=" << params.log_n << " (n=" << params.poly_modulus_degree() << ", slots=" << params.slot_count() << ")\n";
  os << "\t// MaxLevel=" << params.max_level() << ", LogScale=" << params.log_scale << "\n";
  os << "\tparams, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{\n";
  os << "\t\tLogN:            " << params.log_n << ",\n";
  
  // Generate LogQ array
  os << "\t\tLogQ:            []int{";
  for (size_t i = 0; i < params.log_q.size(); ++i) {
    if (i > 0) os << ", ";
    os << params.log_q[i];
  }
  os << "},\n";
  
  // Generate LogP array
  os << "\t\tLogP:            []int{";
  for (size_t i = 0; i < params.log_p.size(); ++i) {
    if (i > 0) os << ", ";
    os << params.log_p[i];
  }
  os << "},\n";
  
  os << "\t\tLogDefaultScale: " << params.log_scale << ",\n";
  os << "\t})\n";
  os << "\tif err != nil {\n";
  os << "\t\tpanic(err)\n";
  os << "\t}\n\n";
  
  os << R"(	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois keys for rotations
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galoisElements, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Encoder, Encryptor, Decryptor, Evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)
)";

  // Add bootstrapper if enabled (Orion-compatible configuration)
  if (params.enable_bootstrap)
  {
    os << "\n\t// Bootstrapper setup (Orion-compatible full configuration)\n";
    os << "\t// This configuration matches Orion's bootstrapping parameters\n";
    os << "\tbtpParamsLit := bootstrapping.ParametersLiteral{\n";
    os << "\t\tLogN: utils.Pointy(params.LogN()),\n";
    
    // LogP for bootstrapping
    os << "\t\tLogP: []int{";
    for (size_t i = 0; i < params.log_p_boot.size(); ++i) {
      if (i > 0) os << ", ";
      os << params.log_p_boot[i];
    }
    os << "},\n";
    
    // Secret key distribution (Hamming weight) - CRITICAL for bootstrap to work
    os << "\t\tXs: ring.Ternary{H: " << params.hamming_weight << "},\n";
    
    // LogSlots - number of slots to bootstrap
    int log_slots = params.effective_log_slots();
    os << "\t\tLogSlots: utils.Pointy(" << log_slots << "),\n";
    
    os << "\t}\n";
    
    os << "\tbtpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParamsLit)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap params error: %v\", err))\n";
    os << "\t}\n";
    
    os << "\n\t// Generate bootstrap evaluation keys\n";
    os << "\tfmt.Println(\"Generating bootstrap keys (this may take a moment)...\")\n";
    os << "\tbtpKeys, _, err := btpParams.GenEvaluationKeys(sk)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap keygen error: %v\", err))\n";
    os << "\t}\n";
    
    os << "\n\t// Create bootstrapper evaluator (global variable)\n";
    os << "\tbootstrapper, err = bootstrapping.NewEvaluator(btpParams, btpKeys)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap evaluator error: %v\", err))\n";
    os << "\t}\n";
    os << "\tfmt.Println(\"Bootstrap keys generated successfully!\")\n";
  }
  
  os << R"(
	// Input/Output maps
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// TODO: Prepare your inputs here
	// Example:
	// values := make([]float64, params.MaxSlots())
	// for i := range values { values[i] = float64(i) }
	// pt := hefloat.NewPlaintext(params, params.MaxLevel())
	// encoder.Encode(values, pt)
	// ct, _ := enc.EncryptNew(pt)
	// encryptedInputs["c0"] = ct

	// Run computation
	)";
  os << func_name;
  os << R"((encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

	// Decrypt and print results
	for name, ct := range encryptedOutputs {
		pt := dec.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		encoder.Decode(pt, values)
		fmt.Printf("%s: [%.4f, %.4f, %.4f, ...]\n", name, values[0], values[1], values[2])
	}

	_ = encodedOutputs
	fmt.Println("CKKS computation completed!")
}
)";
}

} // namespace fheco::code_gen::lattigo

