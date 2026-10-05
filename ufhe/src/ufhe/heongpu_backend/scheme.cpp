#include "ufhe/heongpu_backend/scheme.hpp"

namespace ufhe
{
namespace heongpu_backend
{

Scheme::Scheme(api::scheme_type scheme) : scheme_(scheme) {}

api::scheme_type Scheme::type() const
{
  return scheme_;
}

} // namespace heongpu_backend
} // namespace ufhe
