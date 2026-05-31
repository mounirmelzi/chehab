#include "ufhe/decryptor.hpp"
#include "ufhe/ciphertext.hpp"
#include "ufhe/encryption_context.hpp"
#include "ufhe/plaintext.hpp"
#include "ufhe/seal_backend/decryptor.hpp"
#include "ufhe/seal_backend/encryption_context.hpp"
#include "ufhe/seal_backend/secret_key.hpp"
#include "ufhe/secret_key.hpp"
#ifdef UFHE_USE_HEONGPU
#include "ufhe/heongpu_backend/decryptor.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/secret_key.hpp"
#endif

namespace ufhe
{
Decryptor::Decryptor(const EncryptionContext &context, const SecretKey &secret_key)
{
  if (context.backend() != secret_key.backend())
    throw std::invalid_argument("backend ambiguity, arguments must have the same backend");

  switch (context.backend())
  {
  case api::backend_type::seal:
    underlying_ = std::make_shared<seal_backend::Decryptor>(
      static_cast<const seal_backend::EncryptionContext &>(context.underlying()),
      static_cast<const seal_backend::SecretKey &>(secret_key.underlying()));
    break;

#ifdef UFHE_USE_HEONGPU
  case api::backend_type::heongpu:
    underlying_ = std::make_shared<heongpu_backend::Decryptor>(
      static_cast<const heongpu_backend::EncryptionContext &>(context.underlying()),
      static_cast<const heongpu_backend::SecretKey &>(secret_key.underlying()));
    break;
#endif

  case api::backend_type::none:
    throw std::invalid_argument("no backend is selected");
    break;

  default:
    throw std::invalid_argument("unsupported backend");
    break;
  }
}

void Decryptor::decrypt(const api::Ciphertext &encrypted, api::Plaintext &destination) const
{
  underlying().decrypt(
    safe_static_cast<const Ciphertext &>(encrypted).underlying(),
    *safe_static_cast<Plaintext &>(destination).underlying_);
}

int Decryptor::invariant_noise_budget(const api::Ciphertext &encrypted) const
{
  return underlying().invariant_noise_budget(safe_static_cast<const Ciphertext &>(encrypted).underlying());
}
} // namespace ufhe
