#pragma once

#include "ufhe/api/encryption_context.hpp"
#include <memory>

namespace heongpu
{
class HEContext;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class EncryptionParams;

  class EncryptionContext : public api::EncryptionContext
  {
  public:
    explicit EncryptionContext(const EncryptionParams &params);

    EncryptionContext(const EncryptionContext &copy) = default;
    EncryptionContext &operator=(const EncryptionContext &assign) = default;
    EncryptionContext(EncryptionContext &&source) = default;
    EncryptionContext &operator=(EncryptionContext &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    inline const heongpu::HEContext &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::HEContext> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
