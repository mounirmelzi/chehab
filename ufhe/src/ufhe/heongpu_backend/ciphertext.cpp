#include "ufhe/heongpu_backend/ciphertext.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

Ciphertext::Ciphertext() : underlying_(std::make_shared<heongpu::Ciphertext>()) {}

Ciphertext::Ciphertext(const EncryptionContext &context)
  : underlying_(std::make_shared<heongpu::Ciphertext>(context.underlying()))
{}

Ciphertext::Ciphertext(const Ciphertext &copy)
  : underlying_(std::make_shared<heongpu::Ciphertext>(*copy.underlying_))
{}

Ciphertext &Ciphertext::operator=(const Ciphertext &assign)
{
  underlying_ = std::make_shared<heongpu::Ciphertext>(*assign.underlying_);
  return *this;
}

std::size_t Ciphertext::coeff_modulus_size() const
{
  return underlying_->coeff_modulus_size();
}

std::size_t Ciphertext::poly_modulus_degree() const
{
  return underlying_->ring_size();
}

std::size_t Ciphertext::size() const
{
  return underlying_->size();
}

bool Ciphertext::is_transparent() const
{
  return false;
}

double &Ciphertext::scale() const
{
  return underlying_->scale();
}

} // namespace heongpu_backend
} // namespace ufhe
