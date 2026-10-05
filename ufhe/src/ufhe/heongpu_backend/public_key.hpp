#pragma once

#include "ufhe/api/public_key.hpp"
#include <memory>

namespace heongpu
{
class Publickey;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class PublicKey : public api::PublicKey
  {
    friend class KeyGenerator;

  public:
    PublicKey();

    PublicKey(const PublicKey &copy) = default;
    PublicKey &operator=(const PublicKey &assign) = default;
    PublicKey(PublicKey &&source) = default;
    PublicKey &operator=(PublicKey &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    inline const heongpu::Publickey &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Publickey> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
