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
/**************************/
void fhe_vectorized(int width){
  throw invalid_argument("vectorized implementation doesnt exist");
}
/************************************/
Ciphertext cond(Ciphertext val ,Ciphertext  ch1,Ciphertext ch2){
    return val*ch1 + (1-val)*ch2;
}
void fhe(int slot_count) 
{
  Ciphertext output ;
  Ciphertext c12("c12"); 
  Ciphertext c23("c23"); 
  Ciphertext c13("c13");
  Ciphertext c24("c24"); 
  Ciphertext c14("c14");  
  Ciphertext c34("c34"); 
  // sort 3 
  Ciphertext o123("o123"); 
  Ciphertext o132("o132"); 
  Ciphertext o213("o213");  
  Ciphertext o231("o231");
  Ciphertext o321("o321");  
  Ciphertext o312("o312"); 
  // sort 4 
  Ciphertext o1234("o1234");  
  Ciphertext o1243("o1243");  
  Ciphertext o1324("o1324");  
  Ciphertext o1342("o1342");  
  Ciphertext o1423("o1423");  
  Ciphertext o1432("o1432");  
  Ciphertext o2134("o2134");  
  Ciphertext o2143("o2143");  
  Ciphertext o2314("o2314");  
  Ciphertext o2341("o2341");  
  Ciphertext o2413("o2413");  
  Ciphertext o2431("o2431");  
  Ciphertext o3124("o3124");  
  Ciphertext o3142("o3142");  
  Ciphertext o3214("o3214");  
  Ciphertext o3241("o3241");  
  Ciphertext o3412("o3412");  
  Ciphertext o3421("o3421");  
  Ciphertext o4123("o4123");  
  Ciphertext o4132("o4132");  
  Ciphertext o4213("o4213");  
  Ciphertext o4231("o4231");  
  Ciphertext o4312("o4312");  
  Ciphertext o4321("o4321");

  if(slot_count == 3){
    output = cond(c12, (cond(c23,o123,cond(c13, o132, o312))), (cond(c13,o213,cond(c23, o231, o321))) );
  } else if (slot_count == 4){
    output =  cond(c12,
        cond(c13,
          cond(c23,
            cond(c14,
              cond(c24,
                cond(c34, o1234, o1243),
                cond(c14, o1423, o4123)
              ),
              o1234
            ),
            cond(c14,
              cond(c34,
                cond(c24, o1324, o1342),
                cond(c14, o1432, o4132)
              ),
              o1324
            )
          ),
          cond(c34,
            cond(c14,
              cond(c24, o3124, o3142),
              cond(c34, o3412, o4312)
            ),
            o3124
          )
        ),
        cond(c23,
          cond(c13,
            cond(c24,
              cond(c14,
                cond(c34, o2134, o2143),
                cond(c24, o2413, o4213)
              ),
              o2134
            ),
            cond(c24,
              cond(c34,
                cond(c14, o2314, o2341),
                cond(c24, o2431, o4231)
              ), 
              o2314
            )
          ),
          cond(c34,
            cond(c24,
              cond(c14, o3214, o3241),
              cond(c34, o3421, o4321)
            ),
            o3214
          )
        )
      );
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
    const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
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
    cout<<"Compile time : \n";
    cout << elapsed.count() << " ms\n";
    if (call_quantifier) {
      util::Quantifier quantifier{func};
      quantifier.run_all_analysis();
      quantifier.print_info(cout);
    }
  }
  return 0;
}