"""
Parser for boolean-only Spectra (.spectra) specification files.

Targets the bloemDebugging/ and cimattiAnalyzing/ subsets which use
only boolean variables, no enums/integers/arrays/patterns/imports.

Produces a GR1Formula with propositional sub-formulas using indexed
variable labels (x0, x1, ...) and primed labels (x0', x1', ...) for
next-state references.
"""
import re
from utils.SimpleTree import Formula
from utils.GR1Formula import GR1Formula


def parse_spectra_file(filepath):
    """Parse a .spectra file into a GR1Formula.

    Returns (gr1_formula, metadata) where metadata is a dict with:
        'env_vars': list of env variable names (in declaration order)
        'sys_vars': list of sys variable names (in declaration order)
        'all_vars': list of all variable names (env first, then sys)
        'name_to_index': dict mapping variable name -> index
        'num_vars': total number of boolean variables
    """
    with open(filepath) as f:
        text = f.read()

    # Strip comments
    text = re.sub(r'--[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Phase 1: Extract variable declarations
    env_vars = re.findall(r'env\s+boolean\s+(\w+)\s*;', text)
    sys_vars = re.findall(r'sys\s+boolean\s+(\w+)\s*;', text)
    all_vars = env_vars + sys_vars
    name_to_index = {name: idx for idx, name in enumerate(all_vars)}
    num_vars = len(all_vars)

    # Phase 2: Extract assumption/guarantee blocks
    # Split text into blocks after variable declarations
    # Each block starts with 'assumption' or 'guarantee' (or 'asm'/'gar')
    blocks = re.split(r'\b(assumption|guarantee|asm|gar)\b', text)

    assumptions_init = []
    assumptions_safety = []
    assumptions_justice = []
    guarantees_init = []
    guarantees_safety = []
    guarantees_justice = []

    i = 1  # skip preamble
    while i < len(blocks) - 1:
        kind = blocks[i].strip()
        body = blocks[i + 1].strip()
        i += 2

        # Remove trailing semicolons
        body = body.rstrip(';').strip()
        if not body:
            continue

        is_assumption = kind in ('assumption', 'asm')

        # Classify by temporal prefix
        if body.startswith('GF'):
            # Justice/guarantee: GF(φ) or GF (φ)
            inner = _strip_temporal_prefix(body, 'GF')
            formula = _parse_formula(inner, name_to_index, primed=False)
            if is_assumption:
                assumptions_justice.append(formula)
            else:
                guarantees_justice.append(formula)
        elif body.startswith('G') and not body.startswith('GF'):
            # Safety: G(φ) or G (φ)
            inner = _strip_temporal_prefix(body, 'G')
            formula = _parse_formula(inner, name_to_index, primed=False)
            if is_assumption:
                assumptions_safety.append(formula)
            else:
                guarantees_safety.append(formula)
        else:
            # Init: bare formula
            formula = _parse_formula(body, name_to_index, primed=False)
            if is_assumption:
                assumptions_init.append(formula)
            else:
                guarantees_init.append(formula)

    # Combine multiple init/safety conditions via conjunction
    init_e = _conjoin(assumptions_init)
    init_s = _conjoin(guarantees_init)
    safety_e = _conjoin(assumptions_safety)
    safety_s = _conjoin(guarantees_safety)

    gr1 = GR1Formula(
        justices=assumptions_justice if assumptions_justice else [],
        guarantees=guarantees_justice if guarantees_justice else [],
        init_e=init_e,
        init_s=init_s,
        safety_e=safety_e,
        safety_s=safety_s,
    )

    metadata = {
        'env_vars': env_vars,
        'sys_vars': sys_vars,
        'all_vars': all_vars,
        'name_to_index': name_to_index,
        'num_vars': num_vars,
    }

    return gr1, metadata


def _strip_temporal_prefix(body, prefix):
    """Strip 'G(...)' or 'GF(...)' wrapping, handling optional space."""
    body = body[len(prefix):].strip()
    if body.startswith('(') and body.endswith(')'):
        # Check this is the matching outer paren
        depth = 0
        for j, c in enumerate(body):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                if j == len(body) - 1:
                    return body[1:-1]
                else:
                    break
    return body


def _conjoin(formulas):
    """Conjoin a list of Formula objects. Returns None if empty."""
    if not formulas:
        return None
    result = formulas[0]
    for f in formulas[1:]:
        result = Formula(['&', result, f])
    return result


def _parse_formula(text, name_to_index, primed=False):
    """Parse a propositional formula string into a Formula tree.

    Uses a recursive descent parser to handle:
    - <-> (biconditional, lowest precedence)
    - -> (implication)
    - | (disjunction)
    - & (conjunction)
    - ! (negation)
    - next(expr)
    - var=false, var=true
    - FALSE, TRUE
    - (expr) parenthesized
    - bare variable names
    """
    tokens = _tokenize(text)
    pos = [0]  # mutable position counter
    result = _parse_iff(tokens, pos, name_to_index, primed)
    return result


def _tokenize(text):
    """Tokenize a Spectra formula string."""
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        elif c == '&':
            tokens.append('&')
            i += 1
        elif c == '|':
            tokens.append('|')
            i += 1
        elif c == '!':
            # Check for !=
            if i + 1 < len(text) and text[i + 1] == '=':
                tokens.append('!=')
                i += 2
            else:
                tokens.append('!')
                i += 1
        elif c == '-' and i + 1 < len(text) and text[i + 1] == '>':
            tokens.append('->')
            i += 2
        elif c == '<' and i + 2 < len(text) and text[i + 1:i + 3] == '->':
            tokens.append('<->')
            i += 3
        elif c == '=':
            tokens.append('=')
            i += 1
        elif c == ';':
            i += 1  # skip semicolons
        elif c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append(text[i:j])
            i = j
        elif c.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1  # skip unknown
    return tokens


def _peek(tokens, pos):
    if pos[0] < len(tokens):
        return tokens[pos[0]]
    return None


def _consume(tokens, pos, expected=None):
    tok = tokens[pos[0]] if pos[0] < len(tokens) else None
    if expected is not None and tok != expected:
        raise ValueError("Expected '%s', got '%s' at pos %d" % (expected, tok, pos[0]))
    pos[0] += 1
    return tok


def _parse_iff(tokens, pos, n2i, primed):
    left = _parse_impl(tokens, pos, n2i, primed)
    while _peek(tokens, pos) == '<->':
        _consume(tokens, pos)
        right = _parse_impl(tokens, pos, n2i, primed)
        left = Formula(['<->', left, right])
    return left


def _parse_impl(tokens, pos, n2i, primed):
    left = _parse_or(tokens, pos, n2i, primed)
    while _peek(tokens, pos) == '->':
        _consume(tokens, pos)
        right = _parse_or(tokens, pos, n2i, primed)
        left = Formula(['->', left, right])
    return left


def _parse_or(tokens, pos, n2i, primed):
    left = _parse_and(tokens, pos, n2i, primed)
    while _peek(tokens, pos) == '|':
        _consume(tokens, pos)
        right = _parse_and(tokens, pos, n2i, primed)
        left = Formula(['|', left, right])
    return left


def _parse_and(tokens, pos, n2i, primed):
    left = _parse_not(tokens, pos, n2i, primed)
    while _peek(tokens, pos) == '&':
        _consume(tokens, pos)
        right = _parse_not(tokens, pos, n2i, primed)
        left = Formula(['&', left, right])
    return left


def _parse_not(tokens, pos, n2i, primed):
    if _peek(tokens, pos) == '!':
        _consume(tokens, pos)
        operand = _parse_not(tokens, pos, n2i, primed)
        return Formula(['!', operand])
    return _parse_atom(tokens, pos, n2i, primed)


def _parse_atom(tokens, pos, n2i, primed):
    tok = _peek(tokens, pos)

    if tok == '(':
        _consume(tokens, pos)
        result = _parse_iff(tokens, pos, n2i, primed)
        _consume(tokens, pos, ')')
        return result

    if tok == 'next':
        _consume(tokens, pos)
        _consume(tokens, pos, '(')
        inner = _parse_iff(tokens, pos, n2i, primed=True)
        _consume(tokens, pos, ')')
        return inner

    if tok in ('FALSE', 'false'):
        _consume(tokens, pos)
        return Formula('false')

    if tok in ('TRUE', 'true'):
        _consume(tokens, pos)
        return Formula('true')

    # Variable reference (possibly followed by =true or =false)
    if tok and tok in n2i:
        _consume(tokens, pos)
        var_label = 'x' + str(n2i[tok])
        if primed:
            var_label += "'"

        # Check for =true or =false
        if _peek(tokens, pos) == '=':
            _consume(tokens, pos)
            val = _consume(tokens, pos)
            if val == 'false':
                return Formula(['!', Formula(var_label)])
            else:  # =true
                return Formula(var_label)
        elif _peek(tokens, pos) == '!=':
            _consume(tokens, pos)
            val = _consume(tokens, pos)
            if val == 'true':
                return Formula(['!', Formula(var_label)])
            else:  # !=false
                return Formula(var_label)

        return Formula(var_label)

    raise ValueError("Unexpected token '%s' at pos %d" % (tok, pos[0]))
