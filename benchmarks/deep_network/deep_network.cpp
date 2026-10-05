/**
 * Deep Neural Network Benchmark for CKKS Bootstrapping Test
 * 
 * This benchmark creates a deep circuit that exceeds the available levels,
 * requiring automatic bootstrap placement.
 * 
 * Architecture:
 * - Multiple fully connected layers with polynomial activations
 * - Each layer: MatMul + Polynomial approximation (depth ~2 per layer)
 * - Total depth can exceed available levels
 * 
 * With default parameters (max_level=7), this will require bootstrapping.
 */

#include "fheco/fheco.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;
using namespace fheco;

/**
 * Very deep polynomial chain (for testing bootstrap placement)
 * 
 * Creates: x -> x^2 -> x^4 -> x^8 -> ... -> x^(2^depth)
 * Each squaring adds 1 to multiplicative depth
 */
void fhe_deep_polynomial(int depth)
{
  cout << "Creating deep polynomial chain: depth=" << depth << endl;
  cout << "This will compute x^(2^" << depth << ")" << endl;
  
  Ciphertext x("x");
  Ciphertext result = x;
  
  for (int i = 0; i < depth; i++)
  {
    result = result * result;  // Each squaring adds 1 to mult depth
    cout << "  After squaring " << (i + 1) << ": x^(2^" << (i + 1) << ")" << endl;
  }
  
  result.set_output("result");
}

/**
 * Multi-layer convolution (like a CNN feature extractor)
 * Each layer does: Conv + Activation (x^2)
 */
void fhe_multilayer_conv(int img_size, int kernel_size, int num_layers)
{
  cout << "Creating multi-layer convolution: " << num_layers << " layers" << endl;
  cout << "Image: " << img_size << "x" << img_size << ", Kernel: " << kernel_size << "x" << kernel_size << endl;
  
  int current_size = img_size;
  
  // Initialize image
  vector<vector<Ciphertext>> current_img(current_size, vector<Ciphertext>(current_size));
  for (int i = 0; i < current_size; i++)
  {
    for (int j = 0; j < current_size; j++)
    {
      current_img[i][j] = Ciphertext("img_" + to_string(i) + "_" + to_string(j));
    }
  }
  
  for (int layer = 0; layer < num_layers; layer++)
  {
    // Create kernel for this layer (encrypted)
    vector<vector<Ciphertext>> kernel(kernel_size, vector<Ciphertext>(kernel_size));
    for (int ki = 0; ki < kernel_size; ki++)
    {
      for (int kj = 0; kj < kernel_size; kj++)
      {
        kernel[ki][kj] = Ciphertext("k" + to_string(layer) + "_" + to_string(ki) + "_" + to_string(kj));
      }
    }
    
    int out_size = current_size - kernel_size + 1;
    vector<vector<Ciphertext>> output(out_size, vector<Ciphertext>(out_size));
    
    // Perform convolution
    for (int i = 0; i < out_size; i++)
    {
      for (int j = 0; j < out_size; j++)
      {
        Ciphertext sum = encrypt(0);
        
        for (int ki = 0; ki < kernel_size; ki++)
        {
          for (int kj = 0; kj < kernel_size; kj++)
          {
            sum = sum + (current_img[i + ki][j + kj] * kernel[ki][kj]);
          }
        }
        
        // Apply activation (x^2 - adds 1 to depth)
        output[i][j] = sum * sum;
      }
    }
    
    current_img = output;
    current_size = out_size;
    
    cout << "  Layer " << (layer + 1) << ": " << (current_size + kernel_size - 1) 
         << "x" << (current_size + kernel_size - 1) << " -> " 
         << current_size << "x" << current_size 
         << " (depth += ~" << (kernel_size * kernel_size + 1) << ")" << endl;
  }
  
  // Set outputs
  for (int i = 0; i < current_size; i++)
  {
    for (int j = 0; j < current_size; j++)
    {
      current_img[i][j].set_output("out_" + to_string(i) + "_" + to_string(j));
    }
  }
}

/**
 * Simple deep network: chain of encrypted matrix-vector products
 * Each layer: y = (W * x)^2 where W is encrypted
 */
void fhe_deep_linear_chain(int size, int num_layers)
{
  cout << "Creating deep linear chain: " << num_layers << " layers" << endl;
  cout << "Vector size: " << size << endl;
  
  // Input vector
  vector<Ciphertext> current(size);
  for (int i = 0; i < size; i++)
  {
    current[i] = Ciphertext("x_" + to_string(i));
  }
  
  for (int layer = 0; layer < num_layers; layer++)
  {
    vector<Ciphertext> next(size);
    
    for (int i = 0; i < size; i++)
    {
      // Compute dot product for row i
      Ciphertext sum = encrypt(0);
      for (int j = 0; j < size; j++)
      {
        Ciphertext weight("w" + to_string(layer) + "_" + to_string(i) + "_" + to_string(j));
        sum = sum + (current[j] * weight);
      }
      
      // Apply activation (squaring)
      next[i] = sum * sum;
    }
    
    current = next;
    cout << "  Layer " << (layer + 1) << " completed" << endl;
  }
  
  // Set outputs
  for (int i = 0; i < size; i++)
  {
    current[i].set_output("y_" + to_string(i));
  }
}

void print_usage(const char* program)
{
  cerr << "Usage: " << program << " <mode> [options]" << endl;
  cerr << "Modes:" << endl;
  cerr << "  polynomial <depth>                    # x^(2^depth)" << endl;
  cerr << "  conv <img_size> <kernel_size> <num_layers>" << endl;
  cerr << "  linear <size> <num_layers>" << endl;
  cerr << endl;
  cerr << "Examples:" << endl;
  cerr << "  " << program << " polynomial 10       # x^1024, depth 10" << endl;
  cerr << "  " << program << " conv 8 2 3          # 3-layer conv, 8x8 image, 2x2 kernel" << endl;
  cerr << "  " << program << " linear 4 4          # 4-layer linear, size 4" << endl;
}

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    print_usage(argv[0]);
    return 1;
  }
  
  string mode = argv[1];
  
  try
  {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Enable optimizations
    Compiler::enable_cse();
    Compiler::enable_order_operands();
    Compiler::enable_const_folding();
    
    string func_name;
    
    if (mode == "polynomial")
    {
      if (argc < 3)
      {
        print_usage(argv[0]);
        return 1;
      }
      
      int depth = stoi(argv[2]);
      func_name = "deep_poly";
      
      Compiler::create_func(func_name, 16, 20, false, true);
      fhe_deep_polynomial(depth);
    }
    else if (mode == "conv")
    {
      if (argc < 5)
      {
        print_usage(argv[0]);
        return 1;
      }
      
      int img_size = stoi(argv[2]);
      int kernel_size = stoi(argv[3]);
      int num_layers = stoi(argv[4]);
      func_name = "deep_conv";
      
      int required_slots = img_size * img_size;
      int slot_count = 1;
      while (slot_count < required_slots) slot_count *= 2;
      
      Compiler::create_func(func_name, slot_count, 20, false, true);
      fhe_multilayer_conv(img_size, kernel_size, num_layers);
    }
    else if (mode == "linear")
    {
      if (argc < 4)
      {
        print_usage(argv[0]);
        return 1;
      }
      
      int size = stoi(argv[2]);
      int num_layers = stoi(argv[3]);
      func_name = "deep_linear";
      
      int slot_count = 1;
      while (slot_count < size * size) slot_count *= 2;
      
      Compiler::create_func(func_name, slot_count, 20, false, true);
      fhe_deep_linear_chain(size, num_layers);
    }
    else
    {
      cerr << "Unknown mode: " << mode << endl;
      print_usage(argv[0]);
      return 1;
    }
    
    auto func = Compiler::active_func();
    
    // Apply TRS simplification
    cout << "\n==> Applying TRS simplification..." << endl;
    auto ruleset = Compiler::Ruleset::simplification_ruleset;
    Compiler::compile(func, ruleset, trs::RewriteHeuristic::bottom_up);
    
    // Print statistics
    cout << "\n==> Expression statistics:" << endl;
    util::Quantifier quantifier{func};
    quantifier.run_all_analysis();
    quantifier.print_info(cout);
    
    // Generate Lattigo code with CKKS management
    cout << "\n==> Generating Lattigo code with CKKS management..." << endl;
    
    ofstream go_file("generated_" + func_name + ".go");
    if (!go_file)
    {
      throw runtime_error("Failed to create Go file");
    }
    
    Compiler::gen_lattigo_code(func, go_file, numeric_limits<size_t>::max(), true);
    go_file.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "\n=== Results ===" << endl;
    cout << "Compilation completed in " << duration.count() << " ms" << endl;
    cout << "Generated: generated_" << func_name << ".go" << endl;
    cout << "\nTo run: cd to lattigo_backend && go run generated_" << func_name << ".go" << endl;
  }
  catch (const exception& e)
  {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  
  return 0;
}
