#include "ufhe/heongpu_backend/galois_keys.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

GaloisKeys::GaloisKeys() : underlying_(nullptr) {}

std::size_t GaloisKeys::size() const
{
  return underlying_ ? 1 : 0;
}

} // namespace heongpu_backend
} // namespace ufhe
