#include "ufhe/heongpu_backend/coeff_modulus.hpp"
#include <stdexcept>

namespace ufhe
{
namespace heongpu_backend
{

api::Modulus::vector CoeffModulus::value() const
{
  api::Modulus::vector result;
  result.reserve(moduli_.size());
  for (const Modulus &m : moduli_)
    result.push_back(std::cref(static_cast<const api::Modulus &>(m)));
  return result;
}

std::size_t CoeffModulus::size() const
{
  return moduli_.size();
}

const Modulus &CoeffModulus::at(std::size_t index) const
{
  if (index >= moduli_.size())
    throw std::out_of_range("CoeffModulus index out of range");
  return moduli_[index];
}

CoeffModulus CoeffModulus::BFVDefault(std::size_t poly_modulus_degree)
{
  switch (poly_modulus_degree)
  {
  case 4096:
    return Create(poly_modulus_degree, {36, 36, 37});
  case 8192:
    return Create(poly_modulus_degree, {43, 43, 44, 44, 45});
  case 16384:
    return Create(poly_modulus_degree, {48, 48, 48, 49, 49, 49, 49, 49, 49});
  case 32768:
    return Create(poly_modulus_degree, {55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56});
  default:
    throw std::invalid_argument("unsupported poly_modulus_degree for BFVDefault");
  }
}

CoeffModulus CoeffModulus::Create(std::size_t /*poly_modulus_degree*/, const std::vector<int> &bit_sizes)
{
  CoeffModulus result;
  result.moduli_.reserve(bit_sizes.size());
  for (int bits : bit_sizes)
    result.moduli_.emplace_back(static_cast<std::uint64_t>(1) << (bits - 1));
  return result;
}

} // namespace heongpu_backend
} // namespace ufhe
