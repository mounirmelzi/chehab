#pragma once

#include "ufhe/api/key_generator.hpp"
#include "ufhe/heongpu_backend/secret_key.hpp"
#include <memory>

namespace heongpu
{
class HEKeyGenerator;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class EncryptionContext;

  class KeyGenerator : public api::KeyGenerator
  {
  public:
    explicit KeyGenerator(const EncryptionContext &context);
    KeyGenerator(const EncryptionContext &context, const SecretKey &secret_key);

    KeyGenerator(const KeyGenerator &copy) = default;
    KeyGenerator &operator=(const KeyGenerator &assign) = default;
    KeyGenerator(KeyGenerator &&source) = default;
    KeyGenerator &operator=(KeyGenerator &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    const SecretKey &secret_key() const override;
    void create_public_key(api::PublicKey &destination) const override;
    void create_relin_keys(api::RelinKeys &destination) const override;
    void create_galois_keys(api::GaloisKeys &destination) const override;

    inline const heongpu::HEKeyGenerator &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::HEKeyGenerator> underlying_;
    SecretKey secret_key_;
  };
} // namespace heongpu_backend
} // namespace ufhe
