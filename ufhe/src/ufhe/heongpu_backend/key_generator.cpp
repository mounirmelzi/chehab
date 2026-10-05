#include "ufhe/heongpu_backend/key_generator.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/galois_keys.hpp"
#include "ufhe/heongpu_backend/public_key.hpp"
#include "ufhe/heongpu_backend/relin_keys.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

KeyGenerator::KeyGenerator(const EncryptionContext &context)
  : underlying_(std::make_shared<heongpu::HEKeyGenerator>(context.underlying())),
    context_ref_(std::shared_ptr<const heongpu::HEContext>(
      &context.underlying(), [](const heongpu::HEContext *) {}))
{
  secret_key_.underlying_ = std::make_shared<heongpu::Secretkey>(context.underlying());
  underlying_->generate_secret_key(*secret_key_.underlying_);
}

KeyGenerator::KeyGenerator(const EncryptionContext &context, const SecretKey &secret_key)
  : underlying_(std::make_shared<heongpu::HEKeyGenerator>(context.underlying())),
    context_ref_(std::shared_ptr<const heongpu::HEContext>(
      &context.underlying(), [](const heongpu::HEContext *) {})),
    secret_key_(secret_key)
{}

const SecretKey &KeyGenerator::secret_key() const
{
  return secret_key_;
}

void KeyGenerator::create_public_key(api::PublicKey &destination) const
{
  auto &pk = static_cast<PublicKey &>(destination);
  pk.underlying_ = std::make_shared<heongpu::Publickey>(*context_ref_);
  underlying_->generate_public_key(*pk.underlying_, *secret_key_.underlying_);
}

void KeyGenerator::create_relin_keys(api::RelinKeys &destination) const
{
  auto &rk = static_cast<RelinKeys &>(destination);
  rk.underlying_ = std::make_shared<heongpu::Relinkey>(*context_ref_);
  underlying_->generate_relin_key(*rk.underlying_, *secret_key_.underlying_);
}

void KeyGenerator::create_galois_keys(api::GaloisKeys &destination) const
{
  auto &gk = static_cast<GaloisKeys &>(destination);
  gk.underlying_ = std::make_shared<heongpu::Galoiskey>(*context_ref_);
  underlying_->generate_galois_key(*gk.underlying_, *secret_key_.underlying_);
}

} // namespace heongpu_backend
} // namespace ufhe
