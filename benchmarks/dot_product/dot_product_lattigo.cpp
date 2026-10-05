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

/**
 * Simple dot product for Lattigo CKKS code generation
 * This generates Go code targeting Lattigo library
 */
void fhe_simple(int size)
{
  // Simple dot product: sum(v1[i] * v2[i])
  std::vector<Ciphertext> v1(size);
  std::vector<Ciphertext> v2(size);
  Ciphertext output = encrypt(0);
  
  for (int i = 0; i < size; i++)
  {
    v1[i] = Ciphertext("v1_" + std::to_string(i));
    v2[i] = Ciphertext("v2_" + std::to_string(i));
  }
  
  for (int i = 0; i < size; i++)
  {
    output += (v1[i] * v2[i]);
  }
  
  output.set_output("output");
}

int main(int argc, char **argv)
{
  int slot_count = 4;
  if (argc > 1)
    slot_count = stoi(argv[1]);

  cout << "=== CHEHAB Lattigo Code Generation ===" << endl;
  cout << "Generating CKKS code for dot product with " << slot_count << " elements" << endl;

  // Enable optimizations
  Compiler::enable_cse();
  Compiler::enable_order_operands();
  Compiler::enable_const_folding();

  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  
  string func_name = "fhe";
  t = chrono::high_resolution_clock::now();

  // Create function
  const auto &func = Compiler::create_func(func_name, slot_count, 20, false, true);
  
  // Build the computation
  fhe_simple(slot_count);

  // Apply TRS simplification
  cout << "\n==> Applying TRS simplification..." << endl;
  auto ruleset = Compiler::Ruleset::simplification_ruleset;
  auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
  Compiler::compile(func, ruleset, rewrite_heuristic);

  // Generate Lattigo Go code
  cout << "\n==> Generating Lattigo (Go/CKKS) code..." << endl;
  string gen_name = "generated_" + func_name;
  string gen_path = gen_name + ".go";
  
  ofstream go_os(gen_path);
  if (!go_os)
    throw logic_error("failed to create Go file");

  // Generate Lattigo code with rescale insertion
  Compiler::gen_lattigo_code(func, go_os, numeric_limits<size_t>::max(), true);
  
  go_os.close();

  elapsed = chrono::high_resolution_clock::now() - t;
  cout << "\nCompilation time: " << elapsed.count() << " ms" << endl;

  // Run quantifier for statistics
  cout << "\n==> Expression statistics:" << endl;
  util::Quantifier quantifier{func};
  quantifier.run_all_analysis();
  quantifier.print_info(cout);

  cout << "\n=== Generated file: " << gen_path << " ===" << endl;
  cout << "To compile and run:" << endl;
  cout << "  1. Copy " << gen_path << " to lattigo_backend/" << endl;
  cout << "  2. cd lattigo_backend && go mod tidy && go run " << gen_path << endl;

  return 0;
}

