#pragma once

#include "ufhe/api/secret_key.hpp"
#include <memory>

namespace heongpu
{
class Secretkey;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class SecretKey : public api::SecretKey
  {
    friend class KeyGenerator;

  public:
    SecretKey();

    SecretKey(const SecretKey &copy) = default;
    SecretKey &operator=(const SecretKey &assign) = default;
    SecretKey(SecretKey &&source) = default;
    SecretKey &operator=(SecretKey &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    inline const heongpu::Secretkey &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Secretkey> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
