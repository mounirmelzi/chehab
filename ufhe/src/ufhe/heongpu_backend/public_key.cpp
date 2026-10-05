#include "ufhe/heongpu_backend/public_key.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

PublicKey::PublicKey() : underlying_(nullptr) {}

} // namespace heongpu_backend
} // namespace ufhe
