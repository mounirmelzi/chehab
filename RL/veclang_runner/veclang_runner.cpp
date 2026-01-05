#include "fheco/fheco.hpp"
#include <chrono>
#include <fstream> 
#include <iostream>
#include <string>
#include <vector> 
#include <cmath> 
#include "./global_variables.hpp" 

using namespace std;
using namespace fheco;


int main(int argc, char **argv)
{
  bool cse = true;
  if (argc > 1) cse = stoi(argv[1]);
  if (cse)
  {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
  } 
  else
  {
    Compiler::disable_cse();
    Compiler::disable_order_operands();
  }

  bool const_folding = true; 
  if (argc > 2) const_folding = stoi(argv[2]); 
  if (const_folding)
  {
    Compiler::enable_const_folding();
  }
  else
  {
    Compiler::disable_const_folding();
  }


  string func_name = "fhe";

  const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
  Compiler::format_vectorized_code(func, false);

  string gen_name = "_gen_he_" + func_name;
  string gen_path = "he/" + gen_name;

  ofstream header_os(gen_path + ".hpp");
  if (!header_os) throw logic_error("failed to create header file");

  ofstream source_os(gen_path + ".cpp");
  if (!source_os) throw logic_error("failed to create source file");

  Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os);

  util::Quantifier quantifier{func};
  quantifier.run_all_analysis();
  quantifier.print_info(cout);

  return 0;
}
