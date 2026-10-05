#pragma once

#include "fheco/ir/func.hpp"
#include <memory>

namespace fheco::passes
{
/**
 * @brief Insert rescale operations after cipher-cipher multiplications for CKKS scheme.
 * In CKKS, after each multiplication, the scale doubles and must be rescaled to manage
 * noise growth and prevent overflow.
 * 
 * @param func The function to process
 * @return Number of rescale operations inserted
 */
size_t insert_rescale_after_mul(const std::shared_ptr<ir::Func> &func);

/**
 * @brief Check if a term is a cipher-cipher multiplication (mul or square)
 * that requires a rescale operation in CKKS.
 * 
 * @param term The term to check
 * @return true if the term requires rescale
 */
bool requires_rescale(const ir::Term *term);

} // namespace fheco::passes

