#include "ufhe/heongpu_backend/modulus.hpp"

namespace ufhe
{
namespace heongpu_backend
{

Modulus::Modulus(std::uint64_t value) : value_(value) {}

int Modulus::bit_count() const
{
  std::uint64_t v = value_;
  int count = 0;
  while (v > 0) { v >>= 1; ++count; }
  return count;
}

std::uint64_t Modulus::value() const
{
  return value_;
}

bool Modulus::is_prime() const
{
  if (value_ < 2) return false;
  if (value_ < 4) return true;
  if (value_ % 2 == 0 || value_ % 3 == 0) return false;
  for (std::uint64_t i = 5; i * i <= value_; i += 6)
  {
    if (value_ % i == 0 || value_ % (i + 2) == 0)
      return false;
  }
  return true;
}

bool Modulus::operator==(const api::Modulus &compare) const
{
  return value_ == compare.value();
}

bool Modulus::operator!=(const api::Modulus &compare) const
{
  return value_ != compare.value();
}

bool Modulus::operator<(const api::Modulus &compare) const
{
  return value_ < compare.value();
}

bool Modulus::operator<=(const api::Modulus &compare) const
{
  return value_ <= compare.value();
}

bool Modulus::operator>(const api::Modulus &compare) const
{
  return value_ > compare.value();
}

bool Modulus::operator>=(const api::Modulus &compare) const
{
  return value_ >= compare.value();
}

std::uint64_t Modulus::reduce(std::uint64_t val) const
{
  return (value_ == 0) ? 0 : val % value_;
}

} // namespace heongpu_backend
} // namespace ufhe
