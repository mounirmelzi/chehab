#pragma once

#include "ufhe/api/modulus.hpp"
#include <cstdint>

namespace ufhe
{
namespace heongpu_backend
{
  class Modulus : public api::Modulus
  {
  public:
    explicit Modulus(std::uint64_t value);

    Modulus(const Modulus &copy) = default;
    Modulus &operator=(const Modulus &assign) = default;
    Modulus(Modulus &&source) = default;
    Modulus &operator=(Modulus &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    std::uint64_t value() const override;

  private:
    std::uint64_t value_;
  };
} // namespace heongpu_backend
} // namespace ufhe
