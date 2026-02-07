#include "fheco/fheco.hpp"
#include "fheco/dsl/conv2d_optimized.hpp"

using namespace std;
using namespace fheco;
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Benchmark: Optimized Conv2D for CHEHAB
 * 
 * Compares:
 * 1. Naive approach: Each pixel is separate ciphertext (O(W*W) ciphertexts)
 * 2. Optimized approach: SIMD packing + rotation-based conv (1 ciphertext)
 * 
 * The optimized approach uses the same principle as Orion's Toeplitz-diagonal method:
 * - Pack entire image into single ciphertext (row-major)
 * - Use rotations to align kernel positions
 * - Accumulate weighted contributions
 */

/******************************************************************************
 * NAIVE Conv2D - Each pixel is a separate ciphertext (for comparison)
 ******************************************************************************/
void naive_conv2d(int img_width, int kernel_size) {
    // Each pixel is a separate ciphertext
    vector<vector<Ciphertext>> img(img_width, vector<Ciphertext>(img_width));
    for (int i = 0; i < img_width; i++) {
        for (int j = 0; j < img_width; j++) {
            img[i][j] = Ciphertext("img_" + to_string(i) + "_" + to_string(j));
        }
    }
    
    // Plaintext kernel weights
    vector<vector<Plaintext>> kernel(kernel_size, vector<Plaintext>(kernel_size));
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int kj = 0; kj < kernel_size; kj++) {
            kernel[ki][kj] = Plaintext("k_" + to_string(ki) + "_" + to_string(kj));
        }
    }
    
    // Valid convolution
    int out_width = img_width - kernel_size + 1;
    
    for (int i = 0; i < out_width; i++) {
        for (int j = 0; j < out_width; j++) {
            Plaintext zero_pt(static_cast<integer>(0));
            Ciphertext sum = encrypt(zero_pt);
            
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    Ciphertext product = img[i + ki][j + kj] * kernel[ki][kj];
                    sum = sum + product;
                }
            }
            
            sum.set_output("naive_out_" + to_string(i) + "_" + to_string(j));
        }
    }
}

/******************************************************************************
 * OPTIMIZED Conv2D - SIMD packing + rotation-based
 ******************************************************************************/
void optimized_conv2d(int img_width, int kernel_size) {
    // Single ciphertext containing entire image (SIMD packed, row-major)
    Ciphertext img("packed_image");
    
    // Plaintext kernel weights (example values)
    vector<vector<double>> kernel_weights(kernel_size, vector<double>(kernel_size));
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int kj = 0; kj < kernel_size; kj++) {
            // Use some non-trivial weights
            kernel_weights[ki][kj] = 1.0 / (kernel_size * kernel_size);
        }
    }
    
    // Setup parameters
    Conv2dParams params;
    params.image_height = img_width;
    params.image_width = img_width;
    params.kernel_height = kernel_size;
    params.kernel_width = kernel_size;
    params.num_slots = img_width * img_width;  // At least this many slots
    
    // Use optimized multi-output convolution
    Conv2dMultiOutput conv(params);
    Ciphertext result = conv.execute_all_outputs_plain(img, kernel_weights);
    
    result.set_output("optimized_output");
}

/******************************************************************************
 * DEEP Optimized Conv2D - Multiple layers to test bootstrap
 ******************************************************************************/
void deep_optimized_conv2d(int img_width, int num_layers, int kernel_size) {
    // Single ciphertext containing entire image
    Ciphertext img("packed_image");
    
    int num_slots = img_width * img_width;
    DeepConv2d deep_conv(img_width, img_width, num_slots);
    
    // Add multiple conv layers
    int current_size = img_width;
    for (int layer = 0; layer < num_layers; layer++) {
        DeepConv2d::LayerConfig config;
        config.kernel_height = kernel_size;
        config.kernel_width = kernel_size;
        
        // Initialize weights (averaging filter)
        config.weights.resize(kernel_size, vector<double>(kernel_size));
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                config.weights[ki][kj] = 1.0 / (kernel_size * kernel_size);
            }
        }
        
        deep_conv.add_layer(config);
        current_size = current_size - kernel_size + 1;
        
        if (current_size < kernel_size) {
            cout << "Warning: Output too small for more layers at layer " << layer << endl;
            break;
        }
    }
    
    Ciphertext result = deep_conv.execute(img);
    result.set_output("deep_conv_output");
    
    cout << "Deep conv total multiplicative depth: " << deep_conv.get_total_depth() << endl;
}

/******************************************************************************
 * Run benchmark and generate Lattigo code
 ******************************************************************************/
void run_benchmark(const string& name, void (*func)(int, int), int img_width, int kernel_size) {
    auto start = chrono::high_resolution_clock::now();
    
    // Calculate slot count
    int num_slots = 1;
    int required = img_width * img_width;
    while (num_slots < required) num_slots *= 2;
    
    cout << "\n=== " << name << " ===" << endl;
    cout << "Image: " << img_width << "x" << img_width << ", Kernel: " << kernel_size << "x" << kernel_size << endl;
    cout << "Slots: " << num_slots << endl;
    
    // Create function
    const auto& ir_func = Compiler::create_func(name, num_slots, 20, false, true);
    
    // Build computation
    func(img_width, kernel_size);
    
    // Apply TRS optimization
    auto ruleset = Compiler::Ruleset::simplification_ruleset;
    auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
    Compiler::compile(ir_func, ruleset, rewrite_heuristic);
    
    // Generate Lattigo code
    string go_path = "../lattigo_backend/" + name + ".go";
    ofstream go_os(go_path);
    if (go_os) {
        Compiler::gen_lattigo_code(ir_func, go_os, numeric_limits<size_t>::max(), true);
        go_os.close();
        cout << "Generated: " << go_path << endl;
    }
    
    auto elapsed = chrono::high_resolution_clock::now() - start;
    double ms = chrono::duration<double, milli>(elapsed).count();
    
    // Print statistics
    util::Quantifier quantifier{ir_func};
    quantifier.run_all_analysis();
    
    cout << "\nStatistics:" << endl;
    cout << "  Compilation time: " << fixed << setprecision(2) << ms << " ms" << endl;
    quantifier.print_info(cout);
}

void run_deep_benchmark(const string& name, int img_width, int num_layers, int kernel_size) {
    auto start = chrono::high_resolution_clock::now();
    
    int num_slots = 1;
    int required = img_width * img_width;
    while (num_slots < required) num_slots *= 2;
    num_slots = max(num_slots, 4096);  // Minimum for bootstrap
    
    cout << "\n=== " << name << " ===" << endl;
    cout << "Image: " << img_width << "x" << img_width << ", Layers: " << num_layers << endl;
    cout << "Kernel: " << kernel_size << "x" << kernel_size << ", Slots: " << num_slots << endl;
    
    const auto& ir_func = Compiler::create_func(name, num_slots, 20, false, true);
    
    deep_optimized_conv2d(img_width, num_layers, kernel_size);
    
    auto ruleset = Compiler::Ruleset::simplification_ruleset;
    auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
    Compiler::compile(ir_func, ruleset, rewrite_heuristic);
    
    string go_path = "../lattigo_backend/" + name + ".go";
    ofstream go_os(go_path);
    if (go_os) {
        Compiler::gen_lattigo_code(ir_func, go_os, numeric_limits<size_t>::max(), true);
        go_os.close();
        cout << "Generated: " << go_path << endl;
    }
    
    auto elapsed = chrono::high_resolution_clock::now() - start;
    double ms = chrono::duration<double, milli>(elapsed).count();
    
    util::Quantifier quantifier{ir_func};
    quantifier.run_all_analysis();
    
    cout << "\nStatistics:" << endl;
    cout << "  Compilation time: " << fixed << setprecision(2) << ms << " ms" << endl;
    quantifier.print_info(cout);
}

/******************************************************************************
 * Main
 ******************************************************************************/
int main(int argc, char** argv) {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
    Compiler::enable_const_folding();
    
    int mode = 0;  // 0=all, 1=naive only, 2=optimized only, 3=deep only
    int img_width = 8;
    int kernel_size = 3;
    int num_layers = 5;  // For deep conv
    
    if (argc > 1) mode = stoi(argv[1]);
    if (argc > 2) img_width = stoi(argv[2]);
    if (argc > 3) kernel_size = stoi(argv[3]);
    if (argc > 4) num_layers = stoi(argv[4]);
    
    cout << "============================================" << endl;
    cout << "  CHEHAB Conv2D Optimization Benchmark" << endl;
    cout << "============================================" << endl;
    
    // Comparison table header
    cout << "\nApproach Comparison:" << endl;
    cout << "--------------------" << endl;
    cout << "Naive:     Each pixel = separate ciphertext" << endl;
    cout << "           # ciphertexts = W*W = " << (img_width*img_width) << endl;
    cout << "           # muls = O(W^2 * K^2) = " << (img_width*img_width*kernel_size*kernel_size) << endl;
    cout << endl;
    cout << "Optimized: SIMD packing + rotations (like Orion)" << endl;
    cout << "           # ciphertexts = 1" << endl;
    cout << "           # rotations = K^2 - 1 = " << (kernel_size*kernel_size - 1) << endl;
    cout << "           # muls = K^2 = " << (kernel_size*kernel_size) << endl;
    
    if (mode == 0 || mode == 1) {
        // Only run naive for small sizes (it creates many ciphertexts)
        if (img_width <= 8) {
            run_benchmark("naive_conv2d", naive_conv2d, img_width, kernel_size);
        } else {
            cout << "\nSkipping naive benchmark for large image (would create " 
                 << (img_width*img_width) << " ciphertexts)" << endl;
        }
    }
    
    if (mode == 0 || mode == 2) {
        run_benchmark("optimized_conv2d", optimized_conv2d, img_width, kernel_size);
    }
    
    if (mode == 0 || mode == 3) {
        run_deep_benchmark("deep_conv2d_" + to_string(num_layers) + "layers", 
                          img_width, num_layers, kernel_size);
    }
    
    cout << "\n============================================" << endl;
    cout << "Usage: ./conv2d_optimized_benchmark [mode] [img_width] [kernel_size] [num_layers]" << endl;
    cout << "  mode: 0=all, 1=naive, 2=optimized, 3=deep" << endl;
    cout << "Examples:" << endl;
    cout << "  ./conv2d_optimized_benchmark 2 16 3    # 16x16 image, 3x3 kernel, optimized" << endl;
    cout << "  ./conv2d_optimized_benchmark 3 32 3 10 # 32x32, 10 layers deep" << endl;
    
    return 0;
}
