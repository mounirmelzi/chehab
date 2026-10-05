#pragma once

#include "ufhe/api/plaintext.hpp"
#include <memory>

namespace heongpu
{
class Plaintext;
} // namespace heongpu

namespace ufhe
{
namespace heongpu_backend
{
  class EncryptionContext;

  class Plaintext : public api::Plaintext
  {
    friend class BatchEncoder;
    friend class Encryptor;
    friend class Evaluator;
    friend class Decryptor;

  public:
    Plaintext();
    explicit Plaintext(const EncryptionContext &context);

    Plaintext(const Plaintext &copy);
    Plaintext &operator=(const Plaintext &assign);
    Plaintext(Plaintext &&source) = default;
    Plaintext &operator=(Plaintext &&assign) = default;

    inline api::backend_type backend() const override { return api::backend_type::heongpu; }
    inline api::implementation_level level() const override { return api::implementation_level::low_level; }

    bool is_zero() const override;
    std::string to_string() const override;

    inline const heongpu::Plaintext &underlying() const { return *underlying_; }

  private:
    std::shared_ptr<heongpu::Plaintext> underlying_;
  };
} // namespace heongpu_backend
} // namespace ufhe
