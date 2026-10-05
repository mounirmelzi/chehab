#pragma once

#include "ufhe/api/relin_keys.hpp"
#include <memory>

namespace heongpu
{
class Relinkey;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class RelinKeys : public api::RelinKeys
  {
    friend class KeyGenerator;

  public:
    RelinKeys();

    RelinKeys(const RelinKeys &copy) = default;
    RelinKeys &operator=(const RelinKeys &assign) = default;
    RelinKeys(RelinKeys &&source) = default;
    RelinKeys &operator=(RelinKeys &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    inline const heongpu::Relinkey &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Relinkey> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
