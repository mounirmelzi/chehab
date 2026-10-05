/**
 * Detailed Benchmark for CHEHAB vs Orion Comparison
 * 
 * Measures each compilation phase separately:
 * 1. IR Construction (equivalent to Orion's DAG building)
 * 2. TRS Optimization (equivalent to Orion's polynomial fitting)
 * 3. Bootstrap Placement (same algorithm as Orion)
 * 4. Code Generation
 */

#include "fheco/fheco.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace fheco;

// Timer helper
struct Timer {
    chrono::high_resolution_clock::time_point start;
    string name;
    
    Timer(const string& n) : name(n) {
        start = chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration<double, milli>(end - start);
        return duration.count();
    }
};

/**
 * Deep polynomial: x -> x^(2^depth)
 */
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

/**
 * Multi-layer MLP with quad activations
 * Similar to Orion's DeepQuadNet
 */
void fhe_mlp_quad(int input_size, int hidden_size, int num_layers)
{
    // Input vector
    vector<Ciphertext> x(input_size);
    for (int i = 0; i < input_size; i++)
        x[i] = Ciphertext("x_" + to_string(i));
    
    vector<Ciphertext> current = x;
    
    for (int layer = 0; layer < num_layers; layer++)
    {
        int out_size = (layer == num_layers - 1) ? 1 : hidden_size;
        vector<Ciphertext> output(out_size);
        
        for (int j = 0; j < out_size; j++)
        {
            // Linear: sum of weighted inputs
            Ciphertext sum = Ciphertext("w_" + to_string(layer) + "_" + to_string(j) + "_0") * current[0];
            for (int i = 1; i < (int)current.size(); i++)
            {
                sum = sum + Ciphertext("w_" + to_string(layer) + "_" + to_string(j) + "_" + to_string(i)) * current[i];
            }
            
            // Quad activation: x^2
            output[j] = sum * sum;
        }
        
        current = output;
    }
    
    current[0].set_output("result");
}

/**
 * Convolution layer (simple)
 */
void fhe_conv2d_simple(int img_size, int kernel_size)
{
    int out_size = img_size - kernel_size + 1;
    
    for (int i = 0; i < out_size; i++)
    {
        for (int j = 0; j < out_size; j++)
        {
            Ciphertext sum = encrypt(0);
            
            for (int ki = 0; ki < kernel_size; ki++)
            {
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    Ciphertext img = Ciphertext("img_" + to_string(i+ki) + "_" + to_string(j+kj));
                    Ciphertext w = Ciphertext("w_" + to_string(ki) + "_" + to_string(kj));
                    sum = sum + img * w;
                }
            }
            
            // Quad activation
            sum = sum * sum;
            sum.set_output("out_" + to_string(i) + "_" + to_string(j));
        }
    }
}

void run_benchmark(const string& name, function<void()> build_ir, int depth_hint = 0)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "BENCHMARK: " << name << endl;
    cout << string(70, '=') << endl;
    
    // Phase 1: IR Construction
    Timer t1("IR Construction");
    Compiler::enable_cse();
    Compiler::enable_order_operands();
    Compiler::enable_const_folding();
    
    Compiler::create_func(name, 16, 20, false, true);
    build_ir();
    auto func = Compiler::active_func();
    double time_ir = t1.stop();
    cout << "[1] IR Construction:      " << fixed << setprecision(3) << time_ir << " ms" << endl;
    
    // Phase 2: TRS Optimization
    Timer t2("TRS Optimization");
    auto ruleset = Compiler::Ruleset::simplification_ruleset;
    Compiler::compile(func, ruleset, trs::RewriteHeuristic::bottom_up);
    double time_trs = t2.stop();
    cout << "[2] TRS Optimization:     " << fixed << setprecision(3) << time_trs << " ms" << endl;
    
    // Phase 3: Code Generation (includes bootstrap placement)
    Timer t3("Code Gen + Bootstrap");
    ofstream go_file("/dev/null");  // Discard output for timing
    Compiler::gen_lattigo_code(func, go_file, numeric_limits<size_t>::max(), true);
    double time_codegen = t3.stop();
    cout << "[3] Bootstrap + CodeGen:  " << fixed << setprecision(3) << time_codegen << " ms" << endl;
    
    double total = time_ir + time_trs + time_codegen;
    cout << "----------------------------------------" << endl;
    cout << "TOTAL:                    " << fixed << setprecision(3) << total << " ms" << endl;
}

int main()
{
    cout << "╔════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║           CHEHAB Detailed Compilation Benchmark                    ║" << endl;
    cout << "║   (For fair comparison with Orion)                                 ║" << endl;
    cout << "╚════════════════════════════════════════════════════════════════════╝" << endl;
    
    // Test 1: Deep Polynomial (like Orion's DeepQuadNet)
    run_benchmark("poly_depth4", []{ fhe_deep_polynomial(4); }, 4);
    run_benchmark("poly_depth8", []{ fhe_deep_polynomial(8); }, 8);
    run_benchmark("poly_depth12", []{ fhe_deep_polynomial(12); }, 12);
    run_benchmark("poly_depth16", []{ fhe_deep_polynomial(16); }, 16);
    
    // Test 2: MLP with Quad activation
    run_benchmark("mlp_2layers", []{ fhe_mlp_quad(4, 4, 2); });
    run_benchmark("mlp_4layers", []{ fhe_mlp_quad(4, 4, 4); });
    
    // Test 3: Simple Conv2D
    run_benchmark("conv_4x4_k3", []{ fhe_conv2d_simple(4, 3); });
    
    // Summary table
    cout << "\n" << string(70, '=') << endl;
    cout << "SUMMARY - Compare these with Orion's output" << endl;
    cout << string(70, '=') << endl;
    
    return 0;
}

