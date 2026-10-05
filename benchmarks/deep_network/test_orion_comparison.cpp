/**
 * Test CHEHAB bootstrap placement vs Orion reference
 * 
 * Orion Results (ground truth):
 *   depth=3, levels=5: 0 bootstraps
 *   depth=5, levels=5: 0 bootstraps  
 *   depth=8, levels=5: 1 bootstrap
 *   depth=10, levels=5: 2 bootstraps
 */

#include <fheco/fheco.hpp>
#include <fheco/ckks/ckks_scale_manager.hpp>
#include <fheco/ckks/ckks_params.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace fheco;

void fhe_deep_polynomial(int depth)
{
  Ciphertext x("x");
  Ciphertext result = x;
  
  for (int i = 0; i < depth; i++)
  {
    result = result * result;
  }
  
  result.set_output("result");
}

void test_depth(int depth, int num_levels)
{
  cout << "\n" << string(60, '=') << "\n";
  cout << "Testing depth=" << depth << " with " << num_levels << " levels\n";
  cout << string(60, '=') << "\n";
  
  // Create the function
  string func_name = "test_poly_d" + to_string(depth);
  Compiler::create_func(func_name, 16, 20, false, true);
  
  // Build the polynomial chain
  fhe_deep_polynomial(depth);
  
  auto func = Compiler::active_func();
  
  // Create CKKS params with fixed levels (to match Orion test)
  ckks::CKKSParams params;
  params.log_n = 16;  // Large ring for bootstrap
  params.log_scale = 40;
  
  // LogQ with exactly num_levels multiplication levels
  params.log_q.clear();
  params.log_q.push_back(55);  // First prime
  for (int i = 0; i < num_levels; ++i)
  {
    params.log_q.push_back(40);
  }
  
  // Special primes for key switching
  params.log_p = {61, 61, 61};
  
  // Bootstrap params
  params.log_p_boot = {61, 61, 61, 61, 61, 61, 61, 61};
  params.enable_bootstrap = true;
  
  cout << "CKKS Params:\n";
  cout << "  LogN: " << params.log_n << "\n";
  cout << "  LogQ size: " << params.log_q.size() << " (= " << num_levels << " mult levels)\n";
  cout << "  max_level() = " << params.max_level() << "\n";
  cout << "  Bootstrap enabled: " << (params.enable_bootstrap ? "yes" : "no") << "\n";
  
  // Create scale manager and run bootstrap placement
  cout << "\nRunning bootstrap placement algorithm...\n";
  
  ckks::CKKSScaleManager scale_manager(func, params);
  scale_manager.analyze_and_transform();
  
  size_t num_bootstraps = scale_manager.get_bootstrap_count();
  
  cout << "\n--- CHEHAB Result ---\n";
  cout << "Network requires " << num_bootstraps << " bootstrap operation(s).\n";
  
  // Print expected vs actual (Orion ground truth)
  int expected = 0;
  if (depth == 3 && num_levels == 5) expected = 0;
  if (depth == 5 && num_levels == 5) expected = 0;
  if (depth == 8 && num_levels == 5) expected = 1;
  if (depth == 10 && num_levels == 5) expected = 2;
  
  cout << "Expected (Orion): " << expected << "\n";
  
  if ((int)num_bootstraps == expected)
  {
    cout << "✓ MATCH!\n";
  }
  else
  {
    cout << "✗ MISMATCH! (got " << num_bootstraps << ", expected " << expected << ")\n";
  }
  
  // Clean up for next test
  Compiler::delete_func(func_name);
}

int main(int argc, char* argv[])
{
  cout << "CHEHAB vs Orion Bootstrap Comparison Test\n";
  cout << "==========================================\n";
  
  int num_levels = 5;  // Fixed 5 multiplication levels (to match Orion test)
  
  // Test the same depths as Orion
  vector<int> test_depths = {3, 5, 8, 10};
  
  // Allow command line override
  if (argc > 1)
  {
    test_depths.clear();
    for (int i = 1; i < argc; ++i)
    {
      test_depths.push_back(atoi(argv[i]));
    }
  }
  
  // Enable optimizations
  Compiler::enable_cse();
  Compiler::enable_order_operands();
  Compiler::enable_const_folding();
  
  for (int depth : test_depths)
  {
    test_depth(depth, num_levels);
  }
  
  cout << "\n" << string(60, '=') << "\n";
  cout << "Comparison Summary:\n";
  cout << "| Depth | Levels | Orion | CHEHAB |\n";
  cout << "|-------|--------|-------|--------|\n";
  cout << "|   3   |   5    |   0   |   ?    |\n";
  cout << "|   5   |   5    |   0   |   ?    |\n";
  cout << "|   8   |   5    |   1   |   ?    |\n";
  cout << "|  10   |   5    |   2   |   ?    |\n";
  cout << string(60, '=') << "\n";
  
  return 0;
}
