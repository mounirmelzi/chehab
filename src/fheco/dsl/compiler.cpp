#include "fheco/ckks/ckks_params.hpp"
#include "fheco/ckks/ckks_scale_manager.hpp"
#include "fheco/code_gen/gen_func.hpp"
#include "fheco/code_gen/gen_func_lattigo.hpp"
#include "fheco/code_gen/gen_func_heongpu.hpp"
#include "fheco/dsl/ciphertext.hpp"
#include "fheco/dsl/compiler.hpp"
#include "fheco/dsl/plaintext.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/trs/trs.hpp"
#include "fheco/passes/passes.hpp"
#include "fheco/util/common.hpp"
#include "fheco/util/expr_printer.hpp"
#include "compiler.hpp"
#include <algorithm>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <queue>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "../../../benchmarks/global_variables.cpp"

using namespace std;
using std::queue;
using std::string;
using std::vector;

namespace fheco
{
Compiler::FuncsTable Compiler::funcs_table_{};

Compiler::FuncsTable::const_iterator Compiler::active_func_it_ = Compiler::funcs_table_.cend();

bool Compiler::cse_enabled_ = false;

bool Compiler::order_operands_enabled_ = false;

bool Compiler::const_folding_enabled_ = false;

bool Compiler::scalar_vector_shape_ = true;

bool Compiler::automatic_enc_params_enabled_ = false;

extern "C"
{
  void modify_string(char *str, size_t len);
}

void Compiler::gen_he_code(
  const std::shared_ptr<ir::Func> &func, std::ostream &header_os, std::string_view header_name, std::ostream &source_os,
  size_t rotation_keys_threshold, bool lazy_relin, param_select::EncParams::SecurityLevel security_level)
{
#ifdef FHECO_LOGGING
  clog << "\nrotation_key_selection\n";
#endif
  unordered_set<int> rotation_steps_keys;
  rotation_steps_keys = passes::reduce_rotation_keys(func, rotation_keys_threshold);

#ifdef FHECO_LOGGING
  clog << "\nrelin_insertion\n";
#endif
  size_t relin_keys_count;
  if (lazy_relin)
    relin_keys_count = passes::lazy_relin_heuristic(func);
  else
    relin_keys_count = passes::relin_after_ctxt_ctxt_mul(func);

#ifdef FHECO_LOGGING
  clog << "\ncode_generation\n";
#endif
  code_gen::gen_func(
    func, rotation_steps_keys, header_os, header_name, source_os, security_level, auto_enc_params_selection_enabled());
}

void Compiler::gen_lattigo_code(
  const std::shared_ptr<ir::Func> &func, std::ostream &go_os,
  size_t rotation_keys_threshold, bool insert_rescale_ops)
{
#ifdef FHECO_LOGGING
  clog << "\nLattigo code generation (CKKS)\n";
#endif

  unordered_set<int> rotation_steps_keys;
  rotation_steps_keys = passes::reduce_rotation_keys(func, rotation_keys_threshold);

  std::unordered_map<std::size_t, size_t> term_mul_depth;
  size_t max_mul_depth = 0;
  
  for (auto* term : func->get_top_sorted_terms())
  {
    size_t depth = 0;
    for (auto* operand : term->operands())
    {
      auto it = term_mul_depth.find(operand->id());
      if (it != term_mul_depth.end())
        depth = std::max(depth, it->second);
    }
    
    auto op_type = term->op_code().type();
    if (op_type == ir::OpCode::Type::mul || op_type == ir::OpCode::Type::square)
    {
      bool is_ctxt_ctxt = (op_type == ir::OpCode::Type::square);
      if (!is_ctxt_ctxt && term->operands().size() >= 2)
      {
        auto* op1 = term->operands()[0];
        auto* op2 = term->operands()[1];
        is_ctxt_ctxt = (op1->type() == ir::Term::Type::cipher && 
                        op2->type() == ir::Term::Type::cipher);
      }
      if (is_ctxt_ctxt)
        ++depth;
    }
    
    term_mul_depth[term->id()] = depth;
    max_mul_depth = std::max(max_mul_depth, depth);
  }
  
  const size_t MAX_DEPTH_WITHOUT_BOOTSTRAP = 12;
  bool needs_bootstrap = (max_mul_depth > MAX_DEPTH_WITHOUT_BOOTSTRAP);
  size_t mul_depth = needs_bootstrap ? MAX_DEPTH_WITHOUT_BOOTSTRAP : std::max(max_mul_depth + 1, static_cast<size_t>(3));

  ckks::CKKSParams ckks_params;
  if (needs_bootstrap)
  {
    ckks_params = ckks::CKKSParamSelector::default_params_with_bootstrap(mul_depth);
    ckks_params.enable_bootstrap = true;
  }
  else
  {
    ckks_params = ckks::CKKSParamSelector::default_params(mul_depth);
    ckks_params.enable_bootstrap = false;
  }

  if (insert_rescale_ops)
  {
    ckks::CKKSScaleManager scale_manager(func, ckks_params);
    scale_manager.set_enable_bootstrap(needs_bootstrap);
    scale_manager.analyze_and_transform();
  }

  code_gen::lattigo::gen_func_lattigo(func, rotation_steps_keys, go_os, func->name(), &ckks_params);
}

void Compiler::gen_heongpu_code(
  const std::shared_ptr<ir::Func> &func, std::ostream &cu_os, int scheme,
  size_t rotation_keys_threshold)
{
#ifdef FHECO_LOGGING
  clog << "\nHEonGPU code generation (CUDA)\n";
#endif

  unordered_set<int> rotation_steps_keys;
  rotation_steps_keys = passes::reduce_rotation_keys(func, rotation_keys_threshold);
  passes::relin_after_ctxt_ctxt_mul(func);
  code_gen::heongpu::gen_func_heongpu(func, rotation_steps_keys, cu_os, func->name(), scheme);
}

const shared_ptr<ir::Func> &Compiler::add_func(shared_ptr<ir::Func> func)
{
  if (auto it = funcs_table_.find(func->name()); it != funcs_table_.end())
    throw invalid_argument("function with this name already exists");

  active_func_it_ = funcs_table_.emplace(func->name(), move(func)).first;
  return active_func_it_->second;
}

const shared_ptr<ir::Func> &Compiler::get_func(const string &name)
{
  auto it = funcs_table_.find(name);
  if (it == funcs_table_.end())
    throw invalid_argument("no function with this name was found");

  return it->second;
}

void Compiler::set_active_func(const string &name)
{
  active_func_it_ = funcs_table_.find(name);
  if (active_func_it_ == funcs_table_.cend())
    throw invalid_argument("no function with this name was found");
}

void Compiler::delete_func(const string &name)
{
  if (active_func()->name() == name)
    active_func_it_ = funcs_table_.end();
  funcs_table_.erase(name);
}

ostream &operator<<(ostream &os, Compiler::Ruleset ruleset)
{
  switch (ruleset)
  {
  case Compiler::Ruleset::depth:
    os << "depth";
    break;
  case Compiler::Ruleset::ops_cost:
    os << "ops_cost";
    break;
  case Compiler::Ruleset::joined:
    os << "joined";
    break;
  default:
    throw invalid_argument("invalid ruleset selector");
    break;
  }
  return os;
}

void Compiler::compile(shared_ptr<ir::Func> func, Ruleset ruleset, trs::RewriteHeuristic rewrite_heuristic)
{
  std::cout << "==> greedy TRS part : \n";
  switch (ruleset)
  {
  case Ruleset::simplification_ruleset:
  {
    trs::TRS simplification_ruleset{trs::Ruleset::simplification_ruleset(func)};
    simplification_ruleset.run(rewrite_heuristic);
    break;
  }
  case Ruleset::depth:
  {
    trs::TRS depth_trs{trs::Ruleset::depth_ruleset(func)};
    depth_trs.run(rewrite_heuristic);
    break;
  }
  case Ruleset::ops_cost:
  {
    trs::TRS ops_cost_trs{trs::Ruleset::ops_cost_ruleset(func)};
    ops_cost_trs.run(rewrite_heuristic);
    break;
  }
  case Ruleset::joined:
  {
    trs::TRS joined_trs{trs::Ruleset::joined_ruleset(func)};
    joined_trs.run(rewrite_heuristic);
    break;
  }
  default:
    throw invalid_argument("invalid ruleset selector");
    break;
  }
}

void Compiler::gen_vectorized_code(const std::shared_ptr<ir::Func> &func, int optimization_method, float w_ops, float w_keys, const std::string& framework)
{
  util::ExprPrinter expr_printer(func);
  expr_printer.make_terms_str_expr(util::ExprPrinter::Mode::prefix);
  std::ofstream inputs_file("../inputs.txt");
  std::ofstream expression_file("../expression.txt");
  std::ofstream vectorized_code_file("../vectorized_code.txt");

  if (!inputs_file || !vectorized_code_file || !expression_file)
  {
    std::cerr << "Error opening one of the output files." << std::endl;
    return;
  }

  std::string input_names;
  std::string input_types;
  int vector_width = 0;

  auto process_input_terms = [&](const ir::InputTermsInfo &inputs_info) {
    std::vector<const ir::Term *> input_terms;
    vector<string> prepared_names = {};
    for (const auto &input_info : inputs_info)
    {
      input_terms.push_back(input_info.first);
      prepared_names.push_back(input_info.second.label_);
    }
    std::reverse(prepared_names.begin(), prepared_names.end());
    int comp = 0;
    for (auto it = input_terms.rbegin(); it != input_terms.rend(); ++it)
    {
      input_names += prepared_names[comp] + " ";
      input_types += ((*it)->type() == ir::Term::Type::cipher) ? "1 " : "0 ";
      comp += 1;
    }
  };
  
  process_input_terms(func->data_flow().inputs_info());
  inputs_file << input_names << std::endl;
  inputs_file << input_types << std::endl;

  auto process_output_terms =
    [&](const ir::OutputTermsInfo &outputs_info, const ir::orderedOutputTermsKeys &output_keys) {
      std::vector<const ir::Term *> output_terms;
      for (const auto &output_key : output_keys)
      {
        output_terms.push_back(output_key);
      }
      return output_terms;
    };

  std::vector<const ir::Term *> output_terms =
    process_output_terms(func->data_flow().outputs_info(), func->data_flow().output_keys());
  std::string expression = "(Vec ";
  for (auto it = output_terms.begin(); it != output_terms.end(); ++it)
  {
    string temp_elem = expr_printer.terms_str_exprs().at((*it)->id());
    auto tokens = split(temp_elem);
    temp_elem = constant_folding(tokens);
    expression += temp_elem + " ";
    ++vector_width;
  }

  const char *env_var = std::getenv("VECTOR_WIDTH");
  if (env_var)
  {
    try
    {
      int vector_width_env = std::stoi(env_var);
      while (vector_width < vector_width_env)
      {
        expression += "0 ";
        ++vector_width;
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error parsing VECTOR_WIDTH: " << e.what() << std::endl;
    }
  }
  expression += ")";

  expression_file << expression << std::endl;
  vectorized_code_file << "";
  inputs_file.close();
  expression_file.close();
  vectorized_code_file.close();

  std::cout << "Call the code vectorizer \n";
  call_vectorizer(vector_width, optimization_method, w_ops, w_keys, framework);
  format_vectorized_code(func, false);
}

void Compiler::gen_vectorized_code(const std::shared_ptr<ir::Func> &func, int window, int optimization_method, float w_ops, float w_keys, const std::string& framework)
{
  if (window < 0)
  {
    std::cerr << "Window size must be greater than 0." << std::endl;
    return;
  }
  
  auto rewrite_heuristicc = trs::RewriteHeuristic::bottom_up;
  trs::TRS joined_trs{trs::Ruleset::joined_ruleset(func)};
  joined_trs.run(rewrite_heuristicc);

  int vector_full_width = func->data_flow().output_keys().size();
  int max_vector_size = 4096;

  if (window > max_vector_size) window = max_vector_size;
  if (vector_full_width > max_vector_size && window == 0) window = max_vector_size;

  if (window == 0)
  {
    gen_vectorized_code(func, optimization_method, w_ops, w_keys, framework);
    return;
  }
  else
  {
    util::ExprPrinter expr_printer(func);
    expr_printer.make_terms_str_expr(util::ExprPrinter::Mode::prefix);
    std::ofstream inputs_file("../inputs.txt");
    std::ofstream vectorized_code_file("../vectorized_code.txt");
    if (!inputs_file || !vectorized_code_file)
    {
      std::cerr << "Error opening one of the output files." << std::endl;
      return;
    }

    std::string input_names;
    std::string input_types;
    std::vector<const ir::Term *> input_terms;
    vector<string> prepared_names = {};
    for (const auto &input_info : func->data_flow().inputs_info())
    {
      input_terms.push_back(input_info.first);
      prepared_names.push_back(input_info.second.label_);
    }
    std::reverse(prepared_names.begin(), prepared_names.end());
    int comp = 0;
    for (auto it = input_terms.rbegin(); it != input_terms.rend(); ++it)
    {
      input_names += prepared_names[comp] + " ";
      input_types += ((*it)->type() == ir::Term::Type::cipher) ? "1 " : "0 ";
      comp += 1;
    }
    inputs_file << input_names << std::endl;
    inputs_file << input_types << std::endl;
    inputs_file.close();

    auto process_output_terms =
      [&](const ir::OutputTermsInfo &outputs_info, const ir::orderedOutputTermsKeys &output_keys) {
        std::vector<const ir::Term *> output_terms;
        for (const auto &output_key : output_keys)
        {
          output_terms.push_back(output_key);
        }
        return output_terms;
      };

    std::vector<const ir::Term *> output_terms =
      process_output_terms(func->data_flow().outputs_info(), func->data_flow().output_keys());
    if (vector_full_width < window)
    {
      std::cout << "\nresult vector width smaller than window size ==> windows will be considered=0(deactivated)\n";
      gen_vectorized_code(func, optimization_method, w_ops, w_keys, framework);
      return;
    }
    int index = 0;
    std::string expression = "(Vec ";
    int vector_width = window;
    vectorized_code_file << "";
    vectorized_code_file.close();
    
    vector<int> vector_sizes = {};
    for (auto it = output_terms.begin(); it != output_terms.end(); ++it)
    {
      expression += expr_printer.terms_str_exprs().at((*it)->id()) + " ";
      index = (index + 1) % vector_width;
      if (!index || it == output_terms.end() - 1)
      {
        int current_vector_width = (index == 0) ? vector_width : index;
        for (int i = 0; i < vector_width - current_vector_width; ++i)
        {
          expression += " 0 ";
        }
        expression += " )";
        
        std::ofstream expression_file("../expression.txt");
        if (!expression_file)
        {
          std::cerr << "Error opening expression file." << std::endl;
          return;
        }
        expression_file << expression;
        expression_file.close();
        
        call_vectorizer(vector_width, optimization_method, w_ops, w_keys, framework);

        std::string vectorized_file = "../vectorized_code.txt";
        std::ifstream read_vec_file(vectorized_file);
        std::vector<std::string> expressions;
        std::string expr;
        if (read_vec_file.is_open())
        {
          while (std::getline(read_vec_file, expr))
          {
            expressions.push_back(expr);
          }
          read_vec_file.close();
        }
        int vector_size = std::stoi(expressions.back());
        if (!expressions.empty())
        {
          expressions.pop_back();
        }
        vector_sizes.push_back(vector_size);
        
        std::ofstream write_vec_file(vectorized_file);
        for (auto const &e : expressions)
        {
          write_vec_file << e << "\n";
        }
        write_vec_file.close();
        expression = "(Vec ";
      }
    }

    std::ofstream vectorized_code_file_2("../vectorized_code.txt", std::ios::app);
    if (!vectorized_code_file_2)
    {
      std::cerr << "Error opening vectorized code file." << std::endl;
      return;
    }
    vectorized_code_file_2 << vector_sizes[0];
    vectorized_code_file_2.close();
    format_vectorized_code(func, false);
  }
}

void Compiler::call_vectorizer(int vector_width, int optimization_method, float w_ops, float w_keys, const std::string& framework)
{
  if (optimization_method == 0)
  {
    call_egraph_vectorizer(vector_width, 0);
  }
  else if (optimization_method == 1)
  {
    call_rl_vectorizer(vector_width, w_ops, w_keys, framework);
  }
  else
  {
    std::cerr << "Invalid optimization method specified." << std::endl;
    return;
  }
}

void Compiler::call_egraph_vectorizer(int vector_width, int rewrite_rule_family_index)
{
  string command = "cargo run --release --manifest-path ../../../egraphs/Cargo.toml -- ../expression.txt " +
                   to_string(vector_width) + " " + to_string(rewrite_rule_family_index) + " >> ../vectorized_code.txt";
  system(command.c_str());
}

void Compiler::call_rl_vectorizer(int vector_width, float w_ops, float w_keys, const std::string& framework)
{
  namespace fs = std::filesystem;
  const fs::path original_cwd = fs::current_path();
  const fs::path project_root = fs::absolute("../../../RL");
  const fs::path expr_file = fs::absolute("../expression.txt");
  const fs::path vect_file = fs::absolute("../vectorized_code.txt");

  std::error_code ec;
  fs::current_path(project_root, ec);
  if (ec)
  {
    std::cerr << "Error: cannot change directory to " << project_root << " – " << ec.message() << '\n';
    return;
  }

  std::ostringstream cmd;
  cmd << "python -m fhe_rl ";
  if (!framework.empty()) {
      cmd << "--framework " << framework << " ";
  } else if (w_ops >= 0.0f && w_keys >= 0.0f) {
      cmd << "--framework morl ";
  }
  cmd << "run "
      << "'" << expr_file.string() << "' "
      << "'" << vect_file.string() << "'";
  if (w_ops >= 0.0f && w_keys >= 0.0f) {
      cmd << " --w_ops " << w_ops << " --w_keys " << w_keys;
  }

  std::cout << "Executing: " << cmd.str() << '\n';
  const int rc = std::system(cmd.str().c_str());

  fs::current_path(original_cwd, ec);
  if (ec)
  {
    std::cerr << "Warning: failed to restore working directory – " << ec.message() << '\n';
  }

  if (rc != 0)
  {
    std::cerr << "Vectorizer exited with status " << rc << '\n';
    throw std::runtime_error("Vectorizer failed");
  }
}

// ... [Rest of utility functions: processExpression, update_io_file, vector_constant_folding, build_expression, process_composed_vectors, process, format_vectorized_code remain fully intact]