#include "ufhe/ciphertext.hpp"
#include "ufhe/encryption_context.hpp"
#include "ufhe/seal_backend/ciphertext.hpp"
#include "ufhe/seal_backend/encryption_context.hpp"
#ifdef UFHE_USE_HEONGPU
#include "ufhe/heongpu_backend/ciphertext.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#endif

namespace ufhe
{
Ciphertext::Ciphertext(api::backend_type backend)
{
  switch (backend)
  {
  case api::backend_type::seal:
    underlying_ = std::make_shared<seal_backend::Ciphertext>();
    break;

#ifdef UFHE_USE_HEONGPU
  case api::backend_type::heongpu:
    underlying_ = std::make_shared<heongpu_backend::Ciphertext>();
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

Ciphertext::Ciphertext(const EncryptionContext &context) : underlying_{}
{
  switch (context.backend())
  {
  case api::backend_type::seal:
    underlying_ = std::make_shared<seal_backend::Ciphertext>(
      static_cast<const seal_backend::EncryptionContext &>(context.underlying()));
    break;

#ifdef UFHE_USE_HEONGPU
  case api::backend_type::heongpu:
    underlying_ = std::make_shared<heongpu_backend::Ciphertext>(
      static_cast<const heongpu_backend::EncryptionContext &>(context.underlying()));
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

Ciphertext::Ciphertext(const Ciphertext &copy)
{
  switch (copy.backend())
  {
  case api::backend_type::seal:
    underlying_ =
      std::make_shared<seal_backend::Ciphertext>(static_cast<const seal_backend::Ciphertext &>(copy.underlying()));
    break;

#ifdef UFHE_USE_HEONGPU
  case api::backend_type::heongpu:
    underlying_ =
      std::make_shared<heongpu_backend::Ciphertext>(static_cast<const heongpu_backend::Ciphertext &>(copy.underlying()));
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

Ciphertext &Ciphertext::operator=(const Ciphertext &assign)
{
  switch (assign.backend())
  {
  case api::backend_type::seal:
    underlying_ =
      std::make_shared<seal_backend::Ciphertext>(static_cast<const seal_backend::Ciphertext &>(assign.underlying()));
    return *this;
    break;

#ifdef UFHE_USE_HEONGPU
  case api::backend_type::heongpu:
    underlying_ =
      std::make_shared<heongpu_backend::Ciphertext>(static_cast<const heongpu_backend::Ciphertext &>(assign.underlying()));
    return *this;
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

std::size_t Ciphertext::coeff_modulus_size() const
{
  return underlying().coeff_modulus_size();
}

std::size_t Ciphertext::poly_modulus_degree() const
{
  return underlying().poly_modulus_degree();
}

std::size_t Ciphertext::size() const
{
  return underlying().size();
}

bool Ciphertext::is_transparent() const
{
  return underlying().is_transparent();
}

double &Ciphertext::scale() const
{
  return underlying().scale();
}
} // namespace ufhe
