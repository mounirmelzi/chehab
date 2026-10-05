#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/passes/insert_rescale.hpp"
#include <stdexcept>

using namespace std;

namespace fheco::passes
{

bool requires_rescale(const ir::Term *term)
{
  // In CKKS, cipher-cipher multiplication and square require rescale
  if (term->op_code().type() == ir::OpCode::Type::mul)
  {
    auto arg1 = term->operands()[0];
    auto arg2 = term->operands()[1];
    // Only cipher-cipher multiplication requires rescale
    // cipher-plain multiplication does not double the scale
    if (arg1->type() == ir::Term::Type::cipher && arg2->type() == ir::Term::Type::cipher)
      return true;
    return false;
  }

  if (term->op_code().type() == ir::OpCode::Type::square)
  {
    auto arg = term->operands()[0];
    if (arg->type() == ir::Term::Type::cipher)
      return true;
    return false;
  }

  return false;
}

size_t insert_rescale_after_mul(const shared_ptr<ir::Func> &func)
{
  size_t rescale_count = 0;

  for (auto id : func->get_top_sorted_terms_ids())
  {
    auto term = func->data_flow().get_term(id);
    if (!term)
      continue;

    if (requires_rescale(term))
    {
      // Insert rescale operation after this multiplication
      auto rescale_term = func->insert_op_term(ir::OpCode::rescale, {term});
      // Replace all uses of the multiplication result with the rescaled result
      func->replace_term_with(term, rescale_term);
      ++rescale_count;
    }
  }

  return rescale_count;
}

} // namespace fheco::passes

