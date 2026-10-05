#include "ufhe/heongpu_backend/encryption_params.hpp"
#include "ufhe/heongpu_backend/coeff_modulus.hpp"
#include "ufhe/heongpu_backend/modulus.hpp"
#include "ufhe/heongpu_backend/scheme.hpp"
#include <heongpu/heongpu.cuh>
#include <stdexcept>

namespace ufhe
{
namespace heongpu_backend
{

EncryptionParams::EncryptionParams(const Scheme &scheme)
  : underlying_(std::make_shared<heongpu::Parameters>(
      scheme.type() == api::scheme_type::bfv ? heongpu::scheme_type::bfv : heongpu::scheme_type::ckks)),
    scheme_(std::make_unique<Scheme>(scheme)),
    coeff_modulus_(std::make_unique<CoeffModulus>()),
    plain_modulus_(std::make_unique<Modulus>(0))
{}

EncryptionParams::EncryptionParams(const EncryptionParams &copy)
  : underlying_(std::make_shared<heongpu::Parameters>(*copy.underlying_)),
    scheme_(std::make_unique<Scheme>(*copy.scheme_)),
    coeff_modulus_(std::make_unique<CoeffModulus>(*copy.coeff_modulus_)),
    plain_modulus_(std::make_unique<Modulus>(*copy.plain_modulus_))
{}

EncryptionParams &EncryptionParams::operator=(const EncryptionParams &assign)
{
  underlying_ = std::make_shared<heongpu::Parameters>(*assign.underlying_);
  scheme_ = std::make_unique<Scheme>(*assign.scheme_);
  coeff_modulus_ = std::make_unique<CoeffModulus>(*assign.coeff_modulus_);
  plain_modulus_ = std::make_unique<Modulus>(*assign.plain_modulus_);
  return *this;
}

void EncryptionParams::set_poly_modulus_degree(std::size_t poly_modulus_degree)
{
  underlying_->set_poly_modulus_degree(poly_modulus_degree);
}

void EncryptionParams::set_coeff_modulus(const api::CoeffModulus &coeff_modulus)
{
  const auto &cm = static_cast<const CoeffModulus &>(coeff_modulus);
  coeff_modulus_ = std::make_unique<CoeffModulus>(cm);

  std::vector<int> bit_sizes;
  bit_sizes.reserve(cm.size());
  for (std::size_t i = 0; i < cm.size(); ++i)
  {
    std::uint64_t val = cm.at(i).value();
    int bits = 0;
    while (val > 0) { val >>= 1; ++bits; }
    bit_sizes.push_back(bits);
  }
  underlying_->set_coeff_modulus(bit_sizes);
}

void EncryptionParams::set_plain_modulus(const api::Modulus &plain_modulus)
{
  const auto &m = static_cast<const Modulus &>(plain_modulus);
  plain_modulus_ = std::make_unique<Modulus>(m);
  underlying_->set_plain_modulus(m.value());
}

const Scheme &EncryptionParams::scheme() const
{
  return *scheme_;
}

std::size_t EncryptionParams::poly_modulus_degree() const
{
  return underlying_->poly_modulus_degree();
}

const CoeffModulus &EncryptionParams::coeff_modulus() const
{
  return *coeff_modulus_;
}

const Modulus &EncryptionParams::plain_modulus() const
{
  return *plain_modulus_;
}

} // namespace heongpu_backend
} // namespace ufhe
