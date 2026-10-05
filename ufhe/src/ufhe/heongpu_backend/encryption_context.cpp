#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/encryption_params.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

EncryptionContext::EncryptionContext(const EncryptionParams &params)
  : underlying_(std::make_shared<heongpu::HEContext>(params.underlying()))
{
  underlying_->generate();
}

} // namespace heongpu_backend
} // namespace ufhe
