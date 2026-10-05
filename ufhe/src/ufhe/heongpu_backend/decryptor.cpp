#include "ufhe/heongpu_backend/decryptor.hpp"
#include "ufhe/heongpu_backend/ciphertext.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/plaintext.hpp"
#include "ufhe/heongpu_backend/secret_key.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

Decryptor::Decryptor(const EncryptionContext &context, const SecretKey &secret_key)
  : underlying_(std::make_shared<heongpu::HEDecryptor>(context.underlying(), secret_key.underlying()))
{}

void Decryptor::decrypt(const api::Ciphertext &encrypted, api::Plaintext &destination) const
{
  underlying_->decrypt(
    *static_cast<Plaintext &>(destination).underlying_,
    static_cast<const Ciphertext &>(encrypted).underlying());
}

int Decryptor::invariant_noise_budget(const api::Ciphertext & /*encrypted*/) const
{
  // HEonGPU does not expose noise budget measurement on GPU
  return -1;
}

} // namespace heongpu_backend
} // namespace ufhe
