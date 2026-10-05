#pragma once

#include "fheco/dsl/ciphertext.hpp"
#include "fheco/dsl/plaintext.hpp"
#include "fheco/dsl/compiler.hpp"
#include "fheco/dsl/ops_overloads.hpp"
#include <vector>
#include <cmath>
#include <unordered_map>

namespace fheco {

// Scale factor for converting float weights to integers (CHEHAB uses int64)
// For CKKS, this will be handled by the scale parameter
constexpr int64_t WEIGHT_SCALE = 1000000;  // 10^6

/**
 * Optimized Conv2D using SIMD packing and rotation-based evaluation
 * 
 * Key optimizations:
 * 1. SIMD Packing: Entire image packed into single ciphertext (row-major)
 * 2. Rotation-based convolution: Use rotate + multiply + sum instead of element access
 * 3. Diagonal method: Pre-compute kernel positions as rotation offsets
 * 4. Kernel coefficient merging: Combine same-rotation coefficients
 * 
 * This approach is similar to Orion's Toeplitz-diagonal method but implemented
 * using CHEHAB's existing opcodes (rotate, mul, add).
 */

struct Conv2dParams {
    int image_height;
    int image_width;
    int kernel_height;
    int kernel_width;
    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 0;
    int padding_w = 0;
    int num_slots;  // Total SIMD slots available
};

/**
 * Compute rotation offset for accessing element at (row, col) offset from current position
 * For row-major packing: offset = row * image_width + col
 */
inline int compute_rotation_offset(int row_offset, int col_offset, int image_width) {
    return row_offset * image_width + col_offset;
}

/**
 * Compute output dimensions for valid convolution
 */
inline std::pair<int, int> compute_output_dims(const Conv2dParams& params) {
    int out_h = (params.image_height + 2 * params.padding_h - params.kernel_height) / params.stride_h + 1;
    int out_w = (params.image_width + 2 * params.padding_w - params.kernel_width) / params.stride_w + 1;
    return {out_h, out_w};
}

/**
 * Optimized Conv2D using rotation-based diagonal method
 * 
 * Input: Single ciphertext containing image packed row-major
 * Kernel: Vector of plaintext coefficients (or ciphertext for cipher-cipher)
 * Output: Single ciphertext containing convolution result
 * 
 * Algorithm:
 * 1. For each kernel position (ki, kj):
 *    - Compute rotation offset = ki * image_width + kj
 *    - Rotate input by offset
 *    - Multiply by kernel[ki][kj]
 *    - Accumulate
 * 
 * Optimization: Group kernel positions with same rotation offset (for strided conv)
 */
class Conv2dOptimized {
public:
    Conv2dOptimized(const Conv2dParams& params) : params_(params) {
        precompute_rotation_offsets();
    }
    
    /**
     * Precompute all rotation offsets needed for the convolution
     * This enables merging of coefficients with same rotation
     */
    void precompute_rotation_offsets() {
        rotation_to_kernel_positions_.clear();
        
        for (int ki = 0; ki < params_.kernel_height; ++ki) {
            for (int kj = 0; kj < params_.kernel_width; ++kj) {
                int rotation = compute_rotation_offset(ki, kj, params_.image_width);
                rotation_to_kernel_positions_[rotation].push_back({ki, kj});
            }
        }
        
        num_rotations_ = rotation_to_kernel_positions_.size();
    }
    
/**
 * Execute optimized convolution with plaintext kernel
 * Uses: rotate + cipher-plain mul + add
 * Multiplicative depth: 1 (single mul per element)
 * 
 * Note: weights are provided as doubles but converted to scaled integers
 * for CHEHAB's integer-based API. The CKKS backend will handle proper scaling.
 */
Ciphertext execute_plain_kernel(
    const Ciphertext& input,
    const std::vector<std::vector<double>>& kernel_weights
) {
    Plaintext zero_pt(static_cast<integer>(0));
    Ciphertext result = encrypt(zero_pt);
    bool first = true;
    
    for (const auto& [rotation, positions] : rotation_to_kernel_positions_) {
        // Rotate input to align kernel position (use << operator)
        Ciphertext rotated = (rotation == 0) ? input : (input << rotation);
        
        // Sum all kernel weights for this rotation
        // (For standard conv, each rotation has exactly one kernel position)
        double coeff_sum = 0.0;
        for (const auto& pos : positions) {
            coeff_sum += kernel_weights[pos.first][pos.second];
        }
        
        if (std::abs(coeff_sum) > 1e-10) {
            // Create plaintext for this coefficient (scaled to integer)
            integer scaled_coeff = static_cast<integer>(coeff_sum * WEIGHT_SCALE);
            Plaintext coeff_plain(scaled_coeff);
            
            // Multiply rotated input by coefficient
            Ciphertext term = rotated * coeff_plain;
            
            if (first) {
                result = term;
                first = false;
            } else {
                result = result + term;
            }
        }
    }
    
    return result;
}
    
/**
 * Execute optimized convolution with encrypted kernel
 * Uses: rotate + cipher-cipher mul + add
 * Multiplicative depth: 1 (but requires rescale after each mul)
 */
Ciphertext execute_cipher_kernel(
    const Ciphertext& input,
    const std::vector<std::vector<Ciphertext>>& kernel_ciphers
) {
    Plaintext zero_pt(static_cast<integer>(0));
    Ciphertext result = encrypt(zero_pt);
    bool first = true;
    
    for (int ki = 0; ki < params_.kernel_height; ++ki) {
        for (int kj = 0; kj < params_.kernel_width; ++kj) {
            int rotation = compute_rotation_offset(ki, kj, params_.image_width);
            
            // Rotate input (use << operator)
            Ciphertext rotated = (rotation == 0) ? input : (input << rotation);
            
            // Cipher-cipher multiplication
            Ciphertext term = rotated * kernel_ciphers[ki][kj];
            
            if (first) {
                result = term;
                first = false;
            } else {
                result = result + term;
            }
        }
    }
    
    return result;
}
    
    /**
     * Get statistics about the optimized convolution
     */
    int get_num_rotations() const { return num_rotations_; }
    int get_num_multiplications() const { 
        return params_.kernel_height * params_.kernel_width; 
    }
    int get_multiplicative_depth() const { return 1; }  // Single level of muls
    
    /**
     * Get required rotation keys
     */
    std::vector<int> get_rotation_keys() const {
        std::vector<int> keys;
        for (const auto& [rotation, _] : rotation_to_kernel_positions_) {
            if (rotation != 0) {
                keys.push_back(rotation);
            }
        }
        return keys;
    }

private:
    Conv2dParams params_;
    // Map: rotation_offset -> list of (ki, kj) kernel positions
    std::unordered_map<int, std::vector<std::pair<int, int>>> rotation_to_kernel_positions_;
    int num_rotations_ = 0;
};

/**
 * Multi-output convolution: produces multiple output pixels efficiently
 * 
 * For each output position (oi, oj), we need to shift the base rotation
 * by (oi * stride_h * image_width + oj * stride_w)
 */
class Conv2dMultiOutput {
public:
    Conv2dMultiOutput(const Conv2dParams& params) : params_(params) {
        auto [out_h, out_w] = compute_output_dims(params);
        output_height_ = out_h;
        output_width_ = out_w;
    }
    
/**
 * Execute convolution producing all output pixels packed into one ciphertext
 * 
 * This is the most efficient approach:
 * - Input: image packed row-major in ciphertext
 * - Output: convolution result with outputs at positions corresponding to valid conv
 * 
 * The key insight: for each kernel position, we can compute ALL output pixels
 * with a single rotation, because SIMD applies the same operation to all slots.
 */
Ciphertext execute_all_outputs_plain(
    const Ciphertext& input,
    const std::vector<std::vector<double>>& kernel_weights
) {
    Plaintext zero_pt(static_cast<integer>(0));
    Ciphertext result = encrypt(zero_pt);
    bool first = true;
    
    // For each kernel element
    for (int ki = 0; ki < params_.kernel_height; ++ki) {
        for (int kj = 0; kj < params_.kernel_width; ++kj) {
            double weight = kernel_weights[ki][kj];
            
            if (std::abs(weight) < 1e-10) continue;
            
            // Rotation to align this kernel position
            int rotation = ki * params_.image_width + kj;
            
            // Rotate input (use << operator)
            Ciphertext rotated = (rotation == 0) ? input : (input << rotation);
            
            // Multiply by weight (scaled to integer)
            integer scaled_weight = static_cast<integer>(weight * WEIGHT_SCALE);
            Plaintext weight_plain(scaled_weight);
            Ciphertext term = rotated * weight_plain;
            
            if (first) {
                result = term;
                first = false;
            } else {
                result = result + term;
            }
        }
    }
    
    return result;
}
    
    int get_output_height() const { return output_height_; }
    int get_output_width() const { return output_width_; }

private:
    Conv2dParams params_;
    int output_height_;
    int output_width_;
};

/**
 * Deep convolution: multiple conv layers stacked
 * Tests bootstrap insertion for deep networks
 */
class DeepConv2d {
public:
    struct LayerConfig {
        int kernel_height;
        int kernel_width;
        std::vector<std::vector<double>> weights;  // Plaintext weights
    };
    
    DeepConv2d(int image_height, int image_width, int num_slots)
        : image_height_(image_height), image_width_(image_width), num_slots_(num_slots) {}
    
    void add_layer(const LayerConfig& config) {
        layers_.push_back(config);
    }
    
    /**
     * Execute all layers sequentially
     * Each layer: depth 1, so total depth = num_layers
     */
    Ciphertext execute(const Ciphertext& input) {
        Ciphertext current = input;
        int current_h = image_height_;
        int current_w = image_width_;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];
            
            Conv2dParams params;
            params.image_height = current_h;
            params.image_width = current_w;
            params.kernel_height = layer.kernel_height;
            params.kernel_width = layer.kernel_width;
            params.num_slots = num_slots_;
            
            Conv2dMultiOutput conv(params);
            current = conv.execute_all_outputs_plain(current, layer.weights);
            
            // Update dimensions for next layer (valid convolution)
            current_h = current_h - layer.kernel_height + 1;
            current_w = current_w - layer.kernel_width + 1;
        }
        
        return current;
    }
    
    int get_total_depth() const { return static_cast<int>(layers_.size()); }

private:
    int image_height_;
    int image_width_;
    int num_slots_;
    std::vector<LayerConfig> layers_;
};

} // namespace fheco
