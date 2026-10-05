#include "fheco/fheco.hpp"

using namespace std;
using namespace fheco; 
#include <chrono>
#include <fstream>
#include <iostream>
#include <string> 
#include <vector>
#include <cmath> 
#include "../global_variables.hpp" 
 
/****************/ 
 void fhe_vectorized(int slot_count)
{ 
  Ciphertext c1("v1");
  Ciphertext c2("v2");
  Ciphertext slot_wise_xor = c1 + c2 - 2 * (c1 * c2);
  Ciphertext sum = SumVec(slot_wise_xor,slot_count);
  sum.set_output("output");
}
/************************************/
void fhe(int slot_count)
{
  size_t size = slot_count;
  std::vector<Ciphertext> c1(size);
  std::vector<Ciphertext> c2(size);
  Ciphertext output = encrypt(0);
  for (int i = 0; i < size; i++)
  {
    c1[i] = Ciphertext("v1_" + std::to_string(i));
    c2[i] = Ciphertext("v2_" + std::to_string(i));
  }
  for (int i = 0; i < size; i++)
  {
    output+= c1[i] + c2[i] - 2*(c1[i] * c2[i]);
  }
  output.set_output("output");
}
/******************************************************************************************/
void print_bool_arg(bool arg, const string &name, ostream &os)
{
  os << (arg ? name : "no_" + name);
}

int main(int argc, char **argv)
{
  bool vectorize_code = true;
  if (argc > 1) vectorize_code = stoi(argv[1]);
  
  int slot_count = 1;
  if (argc > 2) slot_count = stoi(argv[2]);

  int arg_idx = 3;
  std::string framework = "constrained";
  if (argc > 3) {
    std::string third_arg = argv[3];
    if (third_arg == "constrained" || third_arg == "mo" || third_arg == "morl" || !isdigit(third_arg[0])) {
      framework = third_arg;
      arg_idx = 4;
    }
  }

  int optimization_method = 0;
  if (argc > arg_idx) optimization_method = stoi(argv[arg_idx]); 

  int window = 0;
  if (argc > arg_idx + 1) window = stoi(argv[arg_idx + 1]);

  bool call_quantifier = true;
  if (argc > arg_idx + 2) call_quantifier = stoi(argv[arg_idx + 2]);

  bool cse = true;
  if (argc > arg_idx + 3) cse = stoi(argv[arg_idx + 3]);
   
  bool const_folding = true; 
  if (argc > arg_idx + 4) const_folding = stoi(argv[arg_idx + 4]);

  int backend = 0;
  if (argc > arg_idx + 5) backend = stoi(argv[arg_idx + 5]);

  float w_ops = 0.5;
  float w_keys = 0.5;
  if (argc > arg_idx + 7) {
    w_ops = stof(argv[arg_idx + 6]);
    w_keys = stof(argv[arg_idx + 7]);
  } else if (argc > arg_idx + 6) {
    w_ops = stof(argv[arg_idx + 6]);
  }

  if (cse) {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
  } else {
    Compiler::disable_cse();
    Compiler::disable_order_operands();
  }

  if (const_folding) Compiler::enable_const_folding();
  else Compiler::disable_const_folding(); 

  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  string func_name = "fhe";
  
  t = chrono::high_resolution_clock::now();
  if (vectorize_code)
  {
    const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
    fhe(slot_count);
    string gen_name = "_gen_he_" + func_name;
    string gen_path = "he/" + gen_name;
    ofstream header_os(gen_path + ".hpp");
    ofstream source_os(gen_path + ".cpp");
    cout << " window is " << window << endl;
    
    if(VECTORIZATION_ENABLED){
      Compiler::gen_vectorized_code(func, window, optimization_method, w_ops, w_keys, framework);  
    }
    if(SIMPLIFICATION_ENABLED){
      auto ruleset = Compiler::Ruleset::depth;
      auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
      Compiler::compile(func, ruleset, rewrite_heuristic);
    }
    if (backend == 0) {
      Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os);
      cout << "Generated SEAL code: " << gen_path << ".hpp/.cpp" << endl;
    } else if (backend == 1) {
      string go_path = "generated_" + func_name + ".go";
      ofstream go_os(go_path);
      Compiler::gen_lattigo_code(func, go_os);
      go_os.close();
      cout << "Generated Lattigo code: " << go_path << endl;
    } else if (backend == 2) {
      string cu_path = "generated_" + func_name + ".cu";
      ofstream cu_os(cu_path);
      Compiler::gen_heongpu_code(func, cu_os, 1);
      cu_os.close();
      cout << "Generated HEonGPU code: " << cu_path << endl;
    }
    elapsed = chrono::high_resolution_clock::now() - t;
    cout << elapsed.count() << " ms\n";
    if (call_quantifier) {
        util::Quantifier quantifier{func};
        quantifier.run_all_analysis();
        quantifier.print_info(cout);
    }
  }
  else
  {
    const auto &func = Compiler::create_func(func_name, slot_count, 20, false, true);
    std::string updated_inputs_file_name = "fhe_io_example_adapted.txt" ;
    std::string inputs_file_name = "fhe_io_example.txt";
    util::copyFile(inputs_file_name,updated_inputs_file_name);
    fhe(slot_count);
    string gen_name = "_gen_he_" + func_name;
    string gen_path = "he/" + gen_name;
    ofstream header_os(gen_path + ".hpp");
    ofstream source_os(gen_path + ".cpp");
    cout << " window is " << window << endl;
    auto ruleset = Compiler::Ruleset::simplification_ruleset;
    auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
    Compiler::compile(func, ruleset, rewrite_heuristic);
    
    if (backend == 0) {
      Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os);
      cout << "Generated SEAL code: " << gen_path << ".hpp/.cpp" << endl;
    } else if (backend == 1) {
      string go_path = "generated_" + func_name + ".go";
      ofstream go_os(go_path);
      Compiler::gen_lattigo_code(func, go_os);
      go_os.close();
      cout << "Generated Lattigo code: " << go_path << endl;
    } else if (backend == 2) {
      string cu_path = "generated_" + func_name + ".cu";
      ofstream cu_os(cu_path);
      Compiler::gen_heongpu_code(func, cu_os, 1);
      cu_os.close();
      cout << "Generated HEonGPU code: " << cu_path << endl;
    }
    elapsed = chrono::high_resolution_clock::now() - t;
    cout << elapsed.count() << " ms\n";
    if (call_quantifier) {
      util::Quantifier quantifier{func};
      quantifier.run_all_analysis();
      quantifier.print_info(cout);
    }
  }
  return 0;
}