#pragma once

#include "ufhe/api/encryption_params.hpp"
#include <memory>

namespace heongpu
{
class Parameters;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class Scheme;
  class CoeffModulus;
  class Modulus;

  class EncryptionParams : public api::EncryptionParams
  {
  public:
    explicit EncryptionParams(const Scheme &scheme);

    EncryptionParams(const EncryptionParams &copy);
    EncryptionParams &operator=(const EncryptionParams &assign);
    EncryptionParams(EncryptionParams &&source) = default;
    EncryptionParams &operator=(EncryptionParams &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    void set_poly_modulus_degree(std::size_t poly_modulus_degree) override;
    void set_coeff_modulus(const api::CoeffModulus &coeff_modulus) override;
    void set_plain_modulus(const api::Modulus &plain_modulus) override;

    const Scheme &scheme() const override;
    std::size_t poly_modulus_degree() const override;
    const CoeffModulus &coeff_modulus() const override;
    const Modulus &plain_modulus() const override;

    inline const heongpu::Parameters &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Parameters> underlying_;
    std::unique_ptr<Scheme> scheme_;
    std::unique_ptr<CoeffModulus> coeff_modulus_;
    std::unique_ptr<Modulus> plain_modulus_;
  };
} // namespace heongpu_backend
} // namespace ufhe
