#include "ufhe/heongpu_backend/evaluator.hpp"
#include "ufhe/heongpu_backend/ciphertext.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

Evaluator::Evaluator(const EncryptionContext &context) : context_(context)
{
  underlying_ = std::make_shared<heongpu::HEOperator>(context.underlying());
}

void Evaluator::negate_inplace(api::Ciphertext &encrypted) const
{
  auto &ct = static_cast<Ciphertext &>(encrypted);
  underlying_->negate(*ct.underlying_);
}

void Evaluator::negate(const api::Ciphertext &encrypted, api::Ciphertext &destination) const
{
  auto &dst = static_cast<Ciphertext &>(destination);
  *dst.underlying_ = static_cast<const Ciphertext &>(encrypted).underlying();
  underlying_->negate(*dst.underlying_);
}

void Evaluator::add_inplace(api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2) const
{
  underlying_->add(
    *static_cast<Ciphertext &>(encrypted1).underlying_,
    static_cast<const Ciphertext &>(encrypted2).underlying());
}

void Evaluator::add(
  const api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2, api::Ciphertext &destination) const
{
  underlying_->add(
    static_cast<const Ciphertext &>(encrypted1).underlying(),
    static_cast<const Ciphertext &>(encrypted2).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::sub_inplace(api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2) const
{
  underlying_->sub(
    *static_cast<Ciphertext &>(encrypted1).underlying_,
    static_cast<const Ciphertext &>(encrypted2).underlying());
}

void Evaluator::sub(
  const api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2, api::Ciphertext &destination) const
{
  underlying_->sub(
    static_cast<const Ciphertext &>(encrypted1).underlying(),
    static_cast<const Ciphertext &>(encrypted2).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::multiply_inplace(api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2) const
{
  underlying_->multiply(
    *static_cast<Ciphertext &>(encrypted1).underlying_,
    static_cast<const Ciphertext &>(encrypted2).underlying());
}

void Evaluator::multiply(
  const api::Ciphertext &encrypted1, const api::Ciphertext &encrypted2, api::Ciphertext &destination) const
{
  underlying_->multiply(
    static_cast<const Ciphertext &>(encrypted1).underlying(),
    static_cast<const Ciphertext &>(encrypted2).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::square_inplace(api::Ciphertext &encrypted) const
{
  auto &ct = static_cast<Ciphertext &>(encrypted);
  underlying_->multiply(*ct.underlying_, ct.underlying());
}

void Evaluator::square(const api::Ciphertext &encrypted, api::Ciphertext &destination) const
{
  auto &src = static_cast<const Ciphertext &>(encrypted);
  underlying_->multiply(src.underlying(), src.underlying(), *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::exponentiate_inplace(
  api::Ciphertext &, std::uint64_t, const api::RelinKeys &) const
{
  throw std::runtime_error("exponentiate not yet supported on HEonGPU backend");
}

void Evaluator::exponentiate(
  const api::Ciphertext &, std::uint64_t, const api::RelinKeys &, api::Ciphertext &) const
{
  throw std::runtime_error("exponentiate not yet supported on HEonGPU backend");
}

void Evaluator::add_plain_inplace(api::Ciphertext &encrypted, const api::Plaintext &plain) const
{
  underlying_->add_plain(*static_cast<Ciphertext &>(encrypted).underlying_,
    static_cast<const Plaintext &>(plain).underlying());
}

void Evaluator::add_plain(
  const api::Ciphertext &encrypted, const api::Plaintext &plain, api::Ciphertext &destination) const
{
  underlying_->add_plain(
    static_cast<const Ciphertext &>(encrypted).underlying(),
    static_cast<const Plaintext &>(plain).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::sub_plain_inplace(api::Ciphertext &encrypted, const api::Plaintext &plain) const
{
  underlying_->sub_plain(*static_cast<Ciphertext &>(encrypted).underlying_,
    static_cast<const Plaintext &>(plain).underlying());
}

void Evaluator::sub_plain(
  const api::Ciphertext &encrypted, const api::Plaintext &plain, api::Ciphertext &destination) const
{
  underlying_->sub_plain(
    static_cast<const Ciphertext &>(encrypted).underlying(),
    static_cast<const Plaintext &>(plain).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::multiply_plain_inplace(api::Ciphertext &encrypted, const api::Plaintext &plain) const
{
  underlying_->multiply_plain(*static_cast<Ciphertext &>(encrypted).underlying_,
    static_cast<const Plaintext &>(plain).underlying());
}

void Evaluator::multiply_plain(
  const api::Ciphertext &encrypted, const api::Plaintext &plain, api::Ciphertext &destination) const
{
  underlying_->multiply_plain(
    static_cast<const Ciphertext &>(encrypted).underlying(),
    static_cast<const Plaintext &>(plain).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::relinearize_inplace(api::Ciphertext &encrypted, const api::RelinKeys &relin_keys) const
{
  underlying_->relinearize(
    *static_cast<Ciphertext &>(encrypted).underlying_,
    static_cast<const RelinKeys &>(relin_keys).underlying());
}

void Evaluator::relinearize(
  const api::Ciphertext &encrypted, const api::RelinKeys &relin_keys, api::Ciphertext &destination) const
{
  underlying_->relinearize(
    static_cast<const Ciphertext &>(encrypted).underlying(),
    static_cast<const RelinKeys &>(relin_keys).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::mod_switch_to_next_inplace(api::Ciphertext &encrypted) const
{
  underlying_->mod_switch(*static_cast<Ciphertext &>(encrypted).underlying_);
}

void Evaluator::mod_switch_to_next(const api::Ciphertext &encrypted, api::Ciphertext &destination) const
{
  *static_cast<Ciphertext &>(destination).underlying_ = static_cast<const Ciphertext &>(encrypted).underlying();
  underlying_->mod_switch(*static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::mod_switch_to_next_inplace(api::Plaintext &) const
{
  throw std::runtime_error("plaintext mod_switch not supported on HEonGPU backend");
}

void Evaluator::mod_switch_to_next(const api::Plaintext &, api::Plaintext &) const
{
  throw std::runtime_error("plaintext mod_switch not supported on HEonGPU backend");
}

void Evaluator::rescale_to_next_inplace(api::Ciphertext &encrypted) const
{
  underlying_->rescale(*static_cast<Ciphertext &>(encrypted).underlying_);
}

void Evaluator::rescale_to_next(const api::Ciphertext &encrypted, api::Ciphertext &destination) const
{
  *static_cast<Ciphertext &>(destination).underlying_ = static_cast<const Ciphertext &>(encrypted).underlying();
  underlying_->rescale(*static_cast<Ciphertext &>(destination).underlying_);
}

void Evaluator::rotate_inplace(api::Ciphertext &encrypted, int steps, const api::GaloisKeys &galois_keys) const
{
  underlying_->rotate(
    *static_cast<Ciphertext &>(encrypted).underlying_, steps,
    static_cast<const GaloisKeys &>(galois_keys).underlying());
}

void Evaluator::rotate(
  const api::Ciphertext &encrypted, int steps, const api::GaloisKeys &galois_keys,
  api::Ciphertext &destination) const
{
  underlying_->rotate(
    static_cast<const Ciphertext &>(encrypted).underlying(), steps,
    static_cast<const GaloisKeys &>(galois_keys).underlying(),
    *static_cast<Ciphertext &>(destination).underlying_);
}

} // namespace heongpu_backend
} // namespace ufhe
