"""
SAT-based trace generation for GR(1) formulas.

Instead of randomly sampling traces and hoping they satisfy complex safety
conditions, this module uses Z3 to *construct* lasso traces that satisfy
specific parts of the GR(1) specification. This produces much more
constraining trace sets that prevent spurious separators.
"""
import sys
sys.setrecursionlimit(10000)

from z3 import Bool, And, Or, Not, Implies, Solver, sat, BoolRef
from utils.Traces import Trace


def _formula_to_z3(formula, vars_at, timestep, trace_length, lasso_start):
    """
    Convert a propositional Formula to a Z3 expression over trace variables.

    vars_at[t][i] is the Z3 Bool for variable i at timestep t.
    Handles primed variables (xi') by looking at the successor position.
    """
    label = formula.label

    if label.startswith('x') and label.endswith("'"):
        var_idx = int(label[1:-1])
        next_t = timestep + 1 if timestep < trace_length - 1 else lasso_start
        return vars_at[next_t][var_idx]
    elif label.startswith('x'):
        var_idx = int(label[1:])
        return vars_at[timestep][var_idx]
    elif label == '&':
        return And(_formula_to_z3(formula.left, vars_at, timestep, trace_length, lasso_start),
                   _formula_to_z3(formula.right, vars_at, timestep, trace_length, lasso_start))
    elif label == '|':
        return Or(_formula_to_z3(formula.left, vars_at, timestep, trace_length, lasso_start),
                  _formula_to_z3(formula.right, vars_at, timestep, trace_length, lasso_start))
    elif label == '!':
        return Not(_formula_to_z3(formula.left, vars_at, timestep, trace_length, lasso_start))
    elif label == '->':
        return Implies(_formula_to_z3(formula.left, vars_at, timestep, trace_length, lasso_start),
                       _formula_to_z3(formula.right, vars_at, timestep, trace_length, lasso_start))
    elif label == '<->':
        l = _formula_to_z3(formula.left, vars_at, timestep, trace_length, lasso_start)
        r = _formula_to_z3(formula.right, vars_at, timestep, trace_length, lasso_start)
        return l == r
    elif label == 'true':
        return True
    elif label == 'false':
        return False
    else:
        raise ValueError("Unknown operator: " + label)


def _encode_assumptions(gr1, vars_at, trace_length, lasso_start):
    """Encode all assumption-side constraints as Z3 expressions."""
    constraints = []
    loop_positions = list(range(lasso_start, trace_length))

    # Init: holds at position 0
    if gr1.init_e is not None:
        constraints.append(_formula_to_z3(gr1.init_e, vars_at, 0, trace_length, lasso_start))

    # Safety: holds at ALL positions
    if gr1.safety_e is not None:
        for t in range(trace_length):
            constraints.append(_formula_to_z3(gr1.safety_e, vars_at, t, trace_length, lasso_start))

    # Justice: each justice holds at SOME loop position
    for j in gr1.justices:
        constraints.append(Or([_formula_to_z3(j, vars_at, t, trace_length, lasso_start)
                               for t in loop_positions]))

    return constraints


def _encode_guarantees(gr1, vars_at, trace_length, lasso_start):
    """Encode all guarantee-side constraints as Z3 expressions."""
    constraints = []
    loop_positions = list(range(lasso_start, trace_length))

    if gr1.init_s is not None:
        constraints.append(_formula_to_z3(gr1.init_s, vars_at, 0, trace_length, lasso_start))

    if gr1.safety_s is not None:
        for t in range(trace_length):
            constraints.append(_formula_to_z3(gr1.safety_s, vars_at, t, trace_length, lasso_start))

    for g in gr1.guarantees:
        constraints.append(Or([_formula_to_z3(g, vars_at, t, trace_length, lasso_start)
                               for t in loop_positions]))

    return constraints


def _encode_guarantee_negation(gr1, vars_at, trace_length, lasso_start):
    """Encode that at least one guarantee fails."""
    negated_parts = []
    loop_positions = list(range(lasso_start, trace_length))

    if gr1.init_s is not None:
        negated_parts.append(
            Not(_formula_to_z3(gr1.init_s, vars_at, 0, trace_length, lasso_start)))

    if gr1.safety_s is not None:
        # Safety fails if it fails at ANY position
        safety_fails = Or([Not(_formula_to_z3(gr1.safety_s, vars_at, t, trace_length, lasso_start))
                           for t in range(trace_length)])
        negated_parts.append(safety_fails)

    for g in gr1.guarantees:
        # Guarantee GFg fails if g holds at NO loop position
        guar_fails = And([Not(_formula_to_z3(g, vars_at, t, trace_length, lasso_start))
                          for t in loop_positions])
        negated_parts.append(guar_fails)

    if not negated_parts:
        return False  # No guarantees to negate
    return Or(negated_parts)


def _extract_trace(model, vars_at, trace_length, lasso_start, num_vars):
    """Extract a Trace from a Z3 model."""
    vec = []
    for t in range(trace_length):
        row = []
        for i in range(num_vars):
            val = model.evaluate(vars_at[t][i], model_completion=True)
            row.append(bool(val))
        vec.append(row)
    return Trace(vec, lasso_start)


def generate_traces_sat(gr1, num_vars, num_pos=15, num_neg=10,
                        trace_lengths=None, lasso_starts=None,
                        max_trials_per_config=None):
    """
    Generate traces using Z3 to construct assumption-satisfying traces.

    Produces four kinds of traces:
    1. Genuine positives: assumptions hold AND guarantees hold
    2. Vacuous positives: assumptions fail (randomly generated, cheap)
    3. Assumption-satisfying negatives: assumptions hold AND guarantees fail
    4. Random negatives (fallback): assumptions fail but classified negative

    Returns (pos_traces, neg_traces) with intendedEvaluation set.
    """
    if trace_lengths is None:
        trace_lengths = [4, 5, 6]
    if lasso_starts is None:
        lasso_starts = None  # will compute per trace_length
    if max_trials_per_config is None:
        # Scale with requested count: need enough per (length, lasso) combo
        max_trials_per_config = max(5, max(num_pos, num_neg) // 2)

    genuine_pos = []
    neg_assum_hold = []

    for length in trace_lengths:
        starts = lasso_starts if lasso_starts else list(range(max(0, length - 4), length))

        for lasso in starts:
            if lasso >= length:
                continue

            # Create Z3 variables: vars_at[t][i] = Bool("v_t_i")
            vars_at = [[Bool("v_%d_%d" % (t, i)) for i in range(num_vars)]
                       for t in range(length)]

            assumption_constraints = _encode_assumptions(gr1, vars_at, length, lasso)
            guarantee_constraints = _encode_guarantees(gr1, vars_at, length, lasso)
            guarantee_negation = _encode_guarantee_negation(gr1, vars_at, length, lasso)

            # --- Genuine positives: assumptions AND guarantees hold ---
            if len(genuine_pos) < num_pos:
                s = Solver()
                for c in assumption_constraints:
                    s.add(c)
                for c in guarantee_constraints:
                    s.add(c)

                for _ in range(max_trials_per_config):
                    if len(genuine_pos) >= num_pos:
                        break
                    if s.check() == sat:
                        m = s.model()
                        tr = _extract_trace(m, vars_at, length, lasso, num_vars)
                        tr.intendedEvaluation = True
                        genuine_pos.append(tr)
                        # Block this solution to get diverse traces
                        block = Or([vars_at[t][i] != m.evaluate(vars_at[t][i], model_completion=True)
                                    for t in range(length) for i in range(num_vars)])
                        s.add(block)
                    else:
                        break

            # --- Assumption-satisfying negatives: assumptions hold, some guarantee fails ---
            if len(neg_assum_hold) < num_neg:
                s = Solver()
                for c in assumption_constraints:
                    s.add(c)
                s.add(guarantee_negation)

                for _ in range(max_trials_per_config):
                    if len(neg_assum_hold) >= num_neg:
                        break
                    if s.check() == sat:
                        m = s.model()
                        tr = _extract_trace(m, vars_at, length, lasso, num_vars)
                        tr.intendedEvaluation = False
                        neg_assum_hold.append(tr)
                        block = Or([vars_at[t][i] != m.evaluate(vars_at[t][i], model_completion=True)
                                    for t in range(length) for i in range(num_vars)])
                        s.add(block)
                    else:
                        break

            if len(genuine_pos) >= num_pos and len(neg_assum_hold) >= num_neg:
                break
        if len(genuine_pos) >= num_pos and len(neg_assum_hold) >= num_neg:
            break

    # --- Vacuous positives: random traces where assumptions fail ---
    import random
    vacuous_pos = []
    for _ in range(5000):
        if len(vacuous_pos) >= num_pos:
            break
        length = random.choice(trace_lengths)
        lasso = random.randint(0, length - 1)
        vec = [[bool(random.randint(0, 1)) for _ in range(num_vars)] for _ in range(length)]
        tr = Trace(vec, lasso)
        if not gr1.evaluate_assumptions(tr):
            tr.intendedEvaluation = True
            vacuous_pos.append(tr)

    # Assemble final trace sets
    # Positives: mix of genuine and vacuous
    half = num_pos // 2
    selected_pos = genuine_pos[:half] + vacuous_pos[:half]
    remaining = num_pos - len(selected_pos)
    if remaining > 0:
        extras = genuine_pos[half:] + vacuous_pos[half:]
        selected_pos += extras[:remaining]

    # Negatives: all assumption-satisfying (the most constraining kind)
    selected_neg = neg_assum_hold[:num_neg]

    return selected_pos, selected_neg
