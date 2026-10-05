#include "ufhe/heongpu_backend/batch_encoder.hpp"
#include "ufhe/heongpu_backend/encryption_context.hpp"
#include "ufhe/heongpu_backend/plaintext.hpp"
#include <heongpu/heongpu.cuh>

namespace ufhe
{
namespace heongpu_backend
{

BatchEncoder::BatchEncoder(const EncryptionContext &context)
  : underlying_(std::make_shared<heongpu::HEEncoder>(context.underlying()))
{}

std::size_t BatchEncoder::slot_count() const
{
  return underlying_->slot_count();
}

void BatchEncoder::encode(const std::vector<std::uint64_t> &values_vector, api::Plaintext &destination) const
{
  auto &pt = static_cast<Plaintext &>(destination);
  underlying_->encode(*pt.underlying_, values_vector);
}

void BatchEncoder::encode(const std::vector<std::int64_t> &values_vector, api::Plaintext &destination) const
{
  std::vector<std::uint64_t> unsigned_values(values_vector.begin(), values_vector.end());
  auto &pt = static_cast<Plaintext &>(destination);
  underlying_->encode(*pt.underlying_, unsigned_values);
}

void BatchEncoder::decode(const api::Plaintext &plain, std::vector<std::uint64_t> &destination) const
{
  underlying_->decode(destination, static_cast<const Plaintext &>(plain).underlying());
}

void BatchEncoder::decode(const api::Plaintext &plain, std::vector<std::int64_t> &destination) const
{
  std::vector<std::uint64_t> unsigned_result;
  underlying_->decode(unsigned_result, static_cast<const Plaintext &>(plain).underlying());
  destination.assign(unsigned_result.begin(), unsigned_result.end());
}

} // namespace heongpu_backend
} // namespace ufhe
