# parser.py

from expr import Const, Var, Op

def parse_sexpr(s):
    """
    Parse a simple S-expression into Expr objects.
    Example:
        "(+ 0 a)" -> Op("+", [Const(0), Var("a")])
        "(Vec (+ a0 b0) (+ a1 b1) ...)" -> Op("Vec", [Op("+", [Var("a0"), Var("b0")]), ...])
    """
    tokens = tokenize(s)
    expr, remaining = parse_tokens(tokens)
    if remaining:
        print(s)
        print(remaining)
        raise SyntaxError("Unexpected tokens remaining")
    return expr

def tokenize(s):
    """
    Tokenize the input string into a list of tokens.
    """
    # Handle parentheses and split by whitespace
    return s.replace('(', ' ( ').replace(')', ' ) ').split()

try:
    from . import config as pytrs_config
except ImportError:
    import config as pytrs_config

def _parse_tokens_constrained(tokens, parent=None):
    """
    Recursively parse tokens into Expr objects.
    Returns a tuple of (Expr, remaining_tokens).
    """
    if not tokens:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        if not tokens:
            raise SyntaxError("Unexpected EOF after '('")
        op = tokens.pop(0)
        args = []

        while tokens and tokens[0] != ')':
            arg, tokens = parse_tokens(tokens, parent=op)
            args.append(arg)
            if not tokens:
                raise SyntaxError("Unexpected EOF, expecting ')'")
        if not tokens:
            raise SyntaxError("Unexpected EOF, expecting ')'")
        tokens.pop(0)  # Remove ')'
        return Op(op, args), tokens
    else:
        try:
            num = float(token)
            if num == int(num) and '.' not in token and 'e' not in token.lower():
                return Const(int(num)), tokens
            return Const(num), tokens
        except ValueError:
            return Var(token, parent), tokens

def _parse_tokens_mo(tokens, parent=None):
    if not tokens:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        if not tokens:
            raise SyntaxError("Unexpected EOF after '('")
        op = tokens.pop(0)
        args = []
        while tokens and tokens[0] != ')':
            arg, tokens = _parse_tokens_mo(tokens, parent=op)
            args.append(arg)
            if not tokens:
                raise SyntaxError("Unexpected EOF, expecting ')'")
        if not tokens:
            raise SyntaxError("Unexpected EOF, expecting ')'")
        tokens.pop(0)
        return Op(op, args), tokens
    elif token.isdigit() or ((token.startswith('-') or token.startswith('+') ) and token[1:].isdigit()):
        return Const(int(token)), tokens
    else:
        return Var(token, parent), tokens

def parse_tokens(tokens, parent=None):
    if getattr(pytrs_config, "framework", "constrained") == "morl":
        return _parse_tokens_mo(tokens, parent)
    return _parse_tokens_constrained(tokens, parent)
