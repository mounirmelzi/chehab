#pragma once

#include "ufhe/api/coeff_modulus.hpp"
#include "ufhe/heongpu_backend/modulus.hpp"
#include <vector>

namespace ufhe
{
namespace heongpu_backend
{
  class CoeffModulus : public api::CoeffModulus
  {
  public:
    CoeffModulus() = default;

    CoeffModulus(const CoeffModulus &copy) = default;
    CoeffModulus &operator=(const CoeffModulus &assign) = default;
    CoeffModulus(CoeffModulus &&source) = default;
    CoeffModulus &operator=(CoeffModulus &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    std::size_t size() const override;
    const Modulus &at(std::size_t index) const override;

    static CoeffModulus BFVDefault(std::size_t poly_modulus_degree);
    static CoeffModulus Create(std::size_t poly_modulus_degree, const std::vector<int> &bit_sizes);

  private:
    std::vector<Modulus> moduli_;
  };
} // namespace heongpu_backend
} // namespace ufhe
