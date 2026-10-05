#pragma once

#include "ufhe/api/scheme.hpp"

namespace ufhe
{
namespace heongpu_backend
{
  class Scheme : public api::Scheme
  {
  public:
    explicit Scheme(api::scheme_type scheme);

    Scheme(const Scheme &copy) = default;
    Scheme &operator=(const Scheme &assign) = default;
    Scheme(Scheme &&source) = default;
    Scheme &operator=(Scheme &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    api::scheme_type scheme() const override;

  private:
    api::scheme_type scheme_;
  };
} // namespace heongpu_backend
} // namespace ufhe
