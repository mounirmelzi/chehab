#include "fheco/fheco.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
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
    Compiler::enable_const_folding();
  else
    Compiler::disable_const_folding();

  // argv[3]: backend selection — 0 = SEAL (default), 1 = Lattigo/CKKS
  int backend = 0;
  if (argc > 3) backend = stoi(argv[3]);

  // Diagnostic: check input files exist and have content
  {
    auto check_file = [](const string &path) {
      ifstream f(path);
      if (!f.is_open()) {
        cerr << "[veclang_runner] MISSING: " << path << endl;
        return;
      }
      f.seekg(0, ios::end);
      auto sz = f.tellg();
      cerr << "[veclang_runner] " << path << " size=" << sz << " bytes" << endl;
    };
    check_file("../vectorized_code.txt");
    check_file("../inputs.txt");
    check_file("fhe_io_example.txt");
  }

  string func_name = "fhe";
  const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
  cerr << "[veclang_runner] create_func done" << endl;
  Compiler::format_vectorized_code(func, false);
  cerr << "[veclang_runner] format_vectorized_code done, terms=" << func->get_top_sorted_terms().size() << endl;

  if (backend == 1)
  {
    cerr << "[veclang_runner] generating Lattigo code..." << endl;
    ofstream go_os("generated_fhe.go");
    if (!go_os) throw logic_error("failed to create Go output file");
    Compiler::gen_lattigo_code(func, go_os, numeric_limits<size_t>::max(), true);
    go_os.close();
    cout << "lattigo_output: generated_fhe.go" << endl;
  }
  else
  {
    string gen_name = "_gen_he_" + func_name;
    string gen_path = "he/" + gen_name;

    ofstream header_os(gen_path + ".hpp");
    if (!header_os) throw logic_error("failed to create header file");

    ofstream source_os(gen_path + ".cpp");
    if (!source_os) throw logic_error("failed to create source file");

    Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os);
  }

  util::Quantifier quantifier{func};
  quantifier.run_all_analysis();
  quantifier.print_info(cout);

  return 0;
}
