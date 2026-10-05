#include "ufhe/heongpu_backend/plaintext.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include <heongpu/heongpu.cuh>
#include <stdexcept>

namespace ufhe
{
namespace heongpu_backend
{

Plaintext::Plaintext() : underlying_(std::make_shared<heongpu::Plaintext>()) {}

Plaintext::Plaintext(const EncryptionContext &context)
  : underlying_(std::make_shared<heongpu::Plaintext>(context.underlying()))
{}

Plaintext::Plaintext(const Plaintext &copy)
  : underlying_(std::make_shared<heongpu::Plaintext>(*copy.underlying_))
{}

Plaintext &Plaintext::operator=(const Plaintext &assign)
{
  underlying_ = std::make_shared<heongpu::Plaintext>(*assign.underlying_);
  return *this;
}

void Plaintext::resize(std::size_t /*coeff_count*/)
{
  throw std::runtime_error("resize not supported on HEonGPU Plaintext");
}

void Plaintext::set_zero(std::size_t /*start_coeff*/, std::size_t /*length*/)
{
  throw std::runtime_error("set_zero not supported on HEonGPU Plaintext");
}

std::size_t Plaintext::capacity() const
{
  return 0;
}

std::size_t Plaintext::coeff_count() const
{
  return 0;
}

std::string Plaintext::to_string() const
{
  return "<heongpu::Plaintext>";
}

bool Plaintext::operator==(const api::Plaintext &compare) const
{
  return underlying_.get() == static_cast<const Plaintext &>(compare).underlying_.get();
}

bool Plaintext::operator!=(const api::Plaintext &compare) const
{
  return !(*this == compare);
}

} // namespace heongpu_backend
} // namespace ufhe
