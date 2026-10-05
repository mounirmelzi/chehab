#pragma once

#include "ufhe/api/galois_keys.hpp"
#include <memory>

namespace heongpu
{
class Galoiskey;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class GaloisKeys : public api::GaloisKeys
  {
    friend class KeyGenerator;

  public:
    GaloisKeys();

    GaloisKeys(const GaloisKeys &copy) = default;
    GaloisKeys &operator=(const GaloisKeys &assign) = default;
    GaloisKeys(GaloisKeys &&source) = default;
    GaloisKeys &operator=(GaloisKeys &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    inline const heongpu::Galoiskey &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Galoiskey> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
