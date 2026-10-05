#include "ufhe/heongpu_backend/relin_keys.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

RelinKeys::RelinKeys() : underlying_(nullptr) {}

std::size_t RelinKeys::size() const
{
  return underlying_ ? 1 : 0;
}

} // namespace heongpu_backend
} // namespace ufhe
