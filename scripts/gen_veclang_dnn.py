#!/usr/bin/env python3
"""Generate VecLang S-expressions for DNN benchmark circuits."""


def dot_product(weights_prefix, inputs, size):
    """Generate S-expr for dot product: sum_i(w_i * x_i)."""
    terms = [f"( * {weights_prefix}_{i} {inputs[i]} )" for i in range(size)]
    return _balanced_sum(terms)


def _balanced_sum(terms):
    """Build a balanced binary addition tree from a list of terms."""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"( + {terms[0]} {terms[1]} )"
    mid = len(terms) // 2
    left = _balanced_sum(terms[:mid])
    right = _balanced_sum(terms[mid:])
    return f"( + {left} {right} )"


def square(expr):
    """Generate S-expr for z*z (x^2 activation)."""
    return f"( * {expr} {expr} )"


def poly_act_deg2(expr):
    """Generate S-expr for 2z^2 + 3z + 1."""
    return f"( + ( + ( * 2 {square(expr)} ) ( * 3 {expr} ) ) 1 )"


def fc_layer(w_prefix, inputs, out_size, in_size):
    """Generate a list of S-exprs for FC layer outputs: out[j] = sum_i(w_j_i * in_i)."""
    outputs = []
    for j in range(out_size):
        outputs.append(dot_product(f"{w_prefix}_{j}", inputs, in_size))
    return outputs


def gen_fc_2layer(size):
    """2-layer MLP: FC(size) -> x^2 -> FC(size) -> x^2, outputs size elements."""
    x = [f"x_{i}" for i in range(size)]
    layer0 = fc_layer("w0", x, size, size)
    act0 = [square(z) for z in layer0]
    layer1 = fc_layer("w1", act0, size, size)
    act1 = [square(h) for h in layer1]
    slots = " ".join(act1)
    return f"(Vec {slots})"


def gen_fc_3layer(size):
    """3-layer MLP: FC -> x^2 -> FC -> x^2 -> FC -> x^2, outputs size elements."""
    x = [f"x_{i}" for i in range(size)]
    layer0 = fc_layer("w0", x, size, size)
    act0 = [square(z) for z in layer0]
    layer1 = fc_layer("w1", act0, size, size)
    act1 = [square(h) for h in layer1]
    layer2 = fc_layer("w2", act1, size, size)
    act2 = [square(r) for r in layer2]
    slots = " ".join(act2)
    return f"(Vec {slots})"


def gen_matmul(size):
    """Pure matrix multiply A*B, outputs size*size elements (row-major)."""
    slots = []
    for i in range(size):
        for j in range(size):
            a_row = [f"a_{i}_{k}" for k in range(size)]
            b_col = [f"b_{k}_{j}" for k in range(size)]
            terms = [f"( * {a_row[k]} {b_col[k]} )" for k in range(size)]
            slots.append(_balanced_sum(terms))
    return f"(Vec {' '.join(slots)})"


def gen_horner_poly(degree, coeff=1):
    """Degree-d polynomial in Horner form: c + x*(c + x*(c + ... + x*c)).

    CKKS multiplicative depth = degree - 1.
    The expression size is O(degree) tokens, making it compact enough for the
    RL agent's TRAE encoder while still reaching high multiplicative depths
    that trigger bootstrap insertion.

    Uses integer coefficients to stay compatible with the TRAE's character-level
    constant tokenizer.
    """
    expr = str(coeff)
    for _ in range(degree):
        expr = f"( + {coeff} ( * x_0 {expr} ) )"
    return f"(Vec {expr})"


if __name__ == "__main__":
    print("Generating VecLang DNN expressions...\n")

    exprs = []

    expr = gen_fc_2layer(8)
    exprs.append((expr, "fc_2layer_8"))
    print(f"fc_2layer_8: {len(expr)} chars")

    expr = gen_fc_3layer(4)
    exprs.append((expr, "fc_3layer_4"))
    print(f"fc_3layer_4: {len(expr)} chars")

    expr = gen_matmul(8)
    exprs.append((expr, "matmul_8x8"))
    print(f"matmul_8x8: {len(expr)} chars")

    expr = gen_horner_poly(14)
    exprs.append((expr, "horner_poly_deg14"))
    print(f"horner_poly_deg14: {len(expr)} chars  (depth 13, triggers 1 bootstrap)")

    expr = gen_horner_poly(26)
    exprs.append((expr, "horner_poly_deg26"))
    print(f"horner_poly_deg26: {len(expr)} chars  (depth 25, triggers 2 bootstraps)")

    for expr_str, name in exprs:
        print(f"\n--- {name} ---")
        print(f"{expr_str[:200]}...")
        print(f"Total length: {len(expr_str)} chars")

    out_lines = [f"{expr_str}:{name}" for expr_str, name in exprs]
    print("\n\nGenerated lines to append to dnn_benchmarks_scaled.txt:")
    for line in out_lines:
        print(f"  {line[:100]}...")
