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
 * Conv2D benchmark for testing CKKS code generation with Lattigo
 * 
 * This benchmark implements a 2D convolution with:
 * - Encrypted input image (cipher-cipher multiplications)
 * - Encrypted kernel weights (to test rescale insertion)
 * - Multiple levels of operations to test level alignment
 */

/******************************************************************************
 * Simple Conv2D (non-vectorized) - tests cipher-cipher multiplication chain
 * 
 * For a 3x3 kernel on a WxW image:
 * output[i][j] = sum(img[i+ki][j+kj] * kernel[ki][kj]) for ki,kj in [-1,1]
 ******************************************************************************/
void fhe_conv2d(int img_width, int kernel_size)
{
  int half_k = kernel_size / 2;
  
  // Input image (encrypted)
  vector<vector<Ciphertext>> img(img_width, vector<Ciphertext>(img_width));
  for (int i = 0; i < img_width; i++)
  {
    for (int j = 0; j < img_width; j++)
    {
      img[i][j] = Ciphertext("img_" + to_string(i) + "_" + to_string(j));
    }
  }
  
  // Kernel weights (encrypted - to force cipher-cipher muls)
  vector<vector<Ciphertext>> kernel(kernel_size, vector<Ciphertext>(kernel_size));
  for (int ki = 0; ki < kernel_size; ki++)
  {
    for (int kj = 0; kj < kernel_size; kj++)
    {
      kernel[ki][kj] = Ciphertext("kernel_" + to_string(ki) + "_" + to_string(kj));
    }
  }
  
  // Output (valid convolution - no padding)
  int out_width = img_width - kernel_size + 1;
  vector<vector<Ciphertext>> output(out_width, vector<Ciphertext>(out_width));
  
  // Perform convolution
  for (int i = 0; i < out_width; i++)
  {
    for (int j = 0; j < out_width; j++)
    {
      Ciphertext sum = encrypt(0);
      
      // Apply kernel
      for (int ki = 0; ki < kernel_size; ki++)
      {
        for (int kj = 0; kj < kernel_size; kj++)
        {
          // Cipher-cipher multiplication (triggers rescale)
          Ciphertext product = img[i + ki][j + kj] * kernel[ki][kj];
          sum += product;
        }
      }
      
      output[i][j] = sum;
    }
  }
  
  // Set outputs
  for (int i = 0; i < out_width; i++)
  {
    for (int j = 0; j < out_width; j++)
    {
      output[i][j].set_output("out_" + to_string(i) + "_" + to_string(j));
    }
  }
}

/******************************************************************************
 * Conv2D with plaintext kernel (cipher-plain muls - no rescale needed)
 ******************************************************************************/
void fhe_conv2d_plain_kernel(int img_width, int kernel_size)
{
  int half_k = kernel_size / 2;
  
  // Input image (encrypted)
  vector<vector<Ciphertext>> img(img_width, vector<Ciphertext>(img_width));
  for (int i = 0; i < img_width; i++)
  {
    for (int j = 0; j < img_width; j++)
    {
      img[i][j] = Ciphertext("img_" + to_string(i) + "_" + to_string(j));
    }
  }
  
  // Kernel weights (plaintext)
  vector<vector<Plaintext>> kernel(kernel_size, vector<Plaintext>(kernel_size));
  for (int ki = 0; ki < kernel_size; ki++)
  {
    for (int kj = 0; kj < kernel_size; kj++)
    {
      kernel[ki][kj] = Plaintext("kernel_" + to_string(ki) + "_" + to_string(kj));
    }
  }
  
  // Output (valid convolution)
  int out_width = img_width - kernel_size + 1;
  
  for (int i = 0; i < out_width; i++)
  {
    for (int j = 0; j < out_width; j++)
    {
      Ciphertext sum = encrypt(0);
      
      for (int ki = 0; ki < kernel_size; ki++)
      {
        for (int kj = 0; kj < kernel_size; kj++)
        {
          // Cipher-plain multiplication (no rescale)
          Ciphertext product = img[i + ki][j + kj] * kernel[ki][kj];
          sum += product;
        }
      }
      
      sum.set_output("out_" + to_string(i) + "_" + to_string(j));
    }
  }
}

/******************************************************************************
 * Multi-layer Conv2D - deeper circuit for stress testing
 ******************************************************************************/
void fhe_conv2d_multilayer(int img_width)
{
  // Small 3x3 kernel for two layers
  int kernel_size = 3;
  
  // Layer 1: Input -> Conv2D -> intermediate
  vector<vector<Ciphertext>> img(img_width, vector<Ciphertext>(img_width));
  for (int i = 0; i < img_width; i++)
  {
    for (int j = 0; j < img_width; j++)
    {
      img[i][j] = Ciphertext("img_" + to_string(i) + "_" + to_string(j));
    }
  }
  
  // First kernel (encrypted)
  vector<vector<Ciphertext>> kernel1(kernel_size, vector<Ciphertext>(kernel_size));
  for (int ki = 0; ki < kernel_size; ki++)
  {
    for (int kj = 0; kj < kernel_size; kj++)
    {
      kernel1[ki][kj] = Ciphertext("k1_" + to_string(ki) + "_" + to_string(kj));
    }
  }
  
  // First convolution
  int mid_width = img_width - kernel_size + 1;
  vector<vector<Ciphertext>> intermediate(mid_width, vector<Ciphertext>(mid_width));
  
  for (int i = 0; i < mid_width; i++)
  {
    for (int j = 0; j < mid_width; j++)
    {
      Ciphertext sum = encrypt(0);
      for (int ki = 0; ki < kernel_size; ki++)
      {
        for (int kj = 0; kj < kernel_size; kj++)
        {
          sum += img[i + ki][j + kj] * kernel1[ki][kj];
        }
      }
      intermediate[i][j] = sum;
    }
  }
  
  // Second kernel (encrypted) - smaller
  vector<vector<Ciphertext>> kernel2(kernel_size, vector<Ciphertext>(kernel_size));
  for (int ki = 0; ki < kernel_size; ki++)
  {
    for (int kj = 0; kj < kernel_size; kj++)
    {
      kernel2[ki][kj] = Ciphertext("k2_" + to_string(ki) + "_" + to_string(kj));
    }
  }
  
  // Second convolution (depth increases!)
  int out_width = mid_width - kernel_size + 1;
  
  for (int i = 0; i < out_width; i++)
  {
    for (int j = 0; j < out_width; j++)
    {
      Ciphertext sum = encrypt(0);
      for (int ki = 0; ki < kernel_size; ki++)
      {
        for (int kj = 0; kj < kernel_size; kj++)
        {
          // This is mul(intermediate, kernel2) where intermediate already has level-1
          // So this tests level alignment!
          sum += intermediate[i + ki][j + kj] * kernel2[ki][kj];
        }
      }
      sum.set_output("out_" + to_string(i) + "_" + to_string(j));
    }
  }
}

/******************************************************************************
 * Main function
 ******************************************************************************/
int main(int argc, char **argv)
{
  // Parse arguments
  int mode = 0;  // 0 = encrypted kernel, 1 = plain kernel, 2 = multilayer
  if (argc > 1)
    mode = stoi(argv[1]);
  
  int img_width = 4;  // Default 4x4 image
  if (argc > 2)
    img_width = stoi(argv[2]);
  
  int kernel_size = 3;  // Default 3x3 kernel
  if (argc > 3)
    kernel_size = stoi(argv[3]);
  
  int backend = 0;  // 0 = SEAL, 1 = Lattigo
  if (argc > 4)
    backend = stoi(argv[4]);
  
  bool call_quantifier = true;
  if (argc > 5)
    call_quantifier = stoi(argv[5]);
  
  // Enable optimizations
  Compiler::enable_cse();
  Compiler::enable_order_operands();
  Compiler::enable_const_folding();
  
  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  
  string func_name = "conv2d";
  
  cout << "=== Conv2D Benchmark ===" << endl;
  cout << "Mode: " << (mode == 0 ? "encrypted kernel" : (mode == 1 ? "plain kernel" : "multilayer")) << endl;
  cout << "Image size: " << img_width << "x" << img_width << endl;
  cout << "Kernel size: " << kernel_size << "x" << kernel_size << endl;
  cout << "Backend: " << (backend == 0 ? "SEAL" : "Lattigo") << endl;
  cout << endl;
  
  t = chrono::high_resolution_clock::now();
  
  // Calculate slot count based on inputs (must be power of 2)
  int num_inputs = img_width * img_width + kernel_size * kernel_size;
  if (mode == 2) {
    num_inputs = img_width * img_width + 2 * kernel_size * kernel_size;
  }
  
  // Round up to next power of 2
  int slot_count = 1;
  while (slot_count < num_inputs)
    slot_count *= 2;
  
  cout << "Slot count: " << slot_count << " (for " << num_inputs << " inputs)" << endl;
  
  const auto &func = Compiler::create_func(func_name, slot_count, 20, false, true);
  
  // Build the computation
  switch (mode)
  {
    case 0:
      cout << "Building Conv2D with encrypted kernel (cipher-cipher muls)..." << endl;
      fhe_conv2d(img_width, kernel_size);
      break;
    case 1:
      cout << "Building Conv2D with plaintext kernel (cipher-plain muls)..." << endl;
      fhe_conv2d_plain_kernel(img_width, kernel_size);
      break;
    case 2:
      cout << "Building multilayer Conv2D (deeper circuit)..." << endl;
      fhe_conv2d_multilayer(img_width);
      break;
  }
  
  // Apply TRS simplification
  cout << "\n==> Applying TRS simplification..." << endl;
  auto ruleset = Compiler::Ruleset::simplification_ruleset;
  auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
  Compiler::compile(func, ruleset, rewrite_heuristic);
  
  // Generate code based on backend
  if (backend == 0)
  {
    // SEAL backend
    string gen_name = "_gen_he_" + func_name;
    string gen_path = "he/" + gen_name;
    ofstream header_os(gen_path + ".hpp");
    ofstream source_os(gen_path + ".cpp");
    
    if (!header_os || !source_os)
      throw logic_error("failed to create SEAL files");
    
    Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os);
    cout << "Generated SEAL code: " << gen_path << ".hpp/.cpp" << endl;
  }
  else
  {
    // Lattigo backend (CKKS)
    string go_path = "generated_" + func_name + ".go";
    ofstream go_os(go_path);
    
    if (!go_os)
      throw logic_error("failed to create Go file");
    
    // Generate with rescale insertion enabled
    Compiler::gen_lattigo_code(func, go_os, numeric_limits<size_t>::max(), true);
    go_os.close();
    
    cout << "Generated Lattigo (CKKS) code: " << go_path << endl;
  }
  
  elapsed = chrono::high_resolution_clock::now() - t;
  cout << "\nCompilation time: " << elapsed.count() << " ms" << endl;
  
  // Run quantifier
  if (call_quantifier)
  {
    cout << "\n==> Expression statistics:" << endl;
    util::Quantifier quantifier{func};
    quantifier.run_all_analysis();
    quantifier.print_info(cout);
  }
  
  // Usage info
  cout << "\n=== Usage ===" << endl;
  cout << "./conv2d [mode] [img_width] [kernel_size] [backend] [quantifier]" << endl;
  cout << "  mode: 0=encrypted kernel, 1=plain kernel, 2=multilayer" << endl;
  cout << "  backend: 0=SEAL, 1=Lattigo" << endl;
  cout << "\nExamples:" << endl;
  cout << "  ./conv2d 0 4 3 1  # 4x4 img, 3x3 kernel, Lattigo" << endl;
  cout << "  ./conv2d 2 6 3 1  # 6x6 img, multilayer, Lattigo" << endl;
  
  return 0;
}

