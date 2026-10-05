#include "ufhe/heongpu_backend/secret_key.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

SecretKey::SecretKey() : underlying_(nullptr) {}

} // namespace heongpu_backend
} // namespace ufhe
