#include "ufhe/heongpu_backend/encryptor.hpp"
#include "ufhe/heongpu_backend/ciphertext.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/plaintext.hpp"
#include "ufhe/heongpu_backend/public_key.hpp"
#include "ufhe/heongpu_backend/secret_key.hpp"
#include <heongpu/heongpu.cuh>
#include <stdexcept>

namespace ufhe
{
namespace heongpu_backend
{

Encryptor::Encryptor(const EncryptionContext &context, const PublicKey &public_key)
  : underlying_(std::make_shared<heongpu::HEEncryptor>(context.underlying(), public_key.underlying()))
{}

Encryptor::Encryptor(const EncryptionContext &context, const SecretKey &secret_key)
  : underlying_(nullptr)
{
  throw std::runtime_error("HEonGPU does not support symmetric encryption via Encryptor(context, secret_key)");
}

Encryptor::Encryptor(
  const EncryptionContext &context, const PublicKey &public_key, const SecretKey & /*secret_key*/)
  : underlying_(std::make_shared<heongpu::HEEncryptor>(context.underlying(), public_key.underlying()))
{}

void Encryptor::encrypt(const api::Plaintext &plain, api::Ciphertext &destination) const
{
  underlying_->encrypt(
    *static_cast<Ciphertext &>(destination).underlying_,
    static_cast<const Plaintext &>(plain).underlying());
}

void Encryptor::encrypt_symmetric(const api::Plaintext & /*plain*/, api::Ciphertext & /*destination*/) const
{
  throw std::runtime_error("symmetric encryption not supported on HEonGPU backend");
}

} // namespace heongpu_backend
} // namespace ufhe
