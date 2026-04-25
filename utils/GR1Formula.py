import re
from utils.SimpleTree import Formula


def _render_next(s):
    """Replace x0', x1', etc. with next(x0), next(x1) in a formula string."""
    return re.sub(r"x(\d+)'", r"next(x\1)", s)


class GR1Formula:
    """
    Represents a GR(1) specification of the form:
    (init_e ∧ □safety_e ∧ □◇J_1 ∧ ... ∧ □◇J_m) → (init_s ∧ □safety_s ∧ □◇G_1 ∧ ... ∧ □◇G_k)

    Each component (init_e, safety_e, J_i, init_s, safety_s, G_j) is a propositional Formula.
    Components can be None if not present.
    """

    def __init__(self, justices=None, guarantees=None,
                 init_e=None, safety_e=None, init_s=None, safety_s=None):
        self.init_e = init_e
        self.safety_e = safety_e
        self.justices = justices if justices is not None else []
        self.init_s = init_s
        self.safety_s = safety_s
        self.guarantees = guarantees if guarantees is not None else []

    def prettyPrint(self):
        assumption_parts = []
        if self.init_e is not None:
            assumption_parts.append(self.init_e.prettyPrint(True))
        if self.safety_e is not None:
            assumption_parts.append('[](' + _render_next(self.safety_e.prettyPrint(True)) + ')')
        for j in self.justices:
            assumption_parts.append('[]<>(' + j.prettyPrint(True) + ')')

        guarantee_parts = []
        if self.init_s is not None:
            guarantee_parts.append(self.init_s.prettyPrint(True))
        if self.safety_s is not None:
            guarantee_parts.append('[](' + _render_next(self.safety_s.prettyPrint(True)) + ')')
        for g in self.guarantees:
            guarantee_parts.append('[]<>(' + g.prettyPrint(True) + ')')

        lhs = ' ∧ '.join(assumption_parts) if assumption_parts else 'true'
        rhs = ' ∧ '.join(guarantee_parts) if guarantee_parts else 'true'

        return '(' + lhs + ') → (' + rhs + ')'

    def evaluate_assumptions(self, trace):
        """
        Evaluate only the assumption side of the GR(1) formula on a trace.
        Returns True if all assumptions (init_e, safety_e, justices) hold.
        """
        loop_positions = list(range(trace.lassoStart, trace.lengthOfTrace))

        if self.init_e is not None:
            if not _eval_prop(self.init_e, trace, 0):
                return False

        if self.safety_e is not None:
            for t in range(trace.lengthOfTrace):
                if not _eval_prop(self.safety_e, trace, t):
                    return False

        for j in self.justices:
            if not any(_eval_prop(j, trace, t) for t in loop_positions):
                return False

        return True

    def evaluate_guarantees(self, trace):
        """
        Evaluate only the guarantee side of the GR(1) formula on a trace.
        Returns True if all guarantees (init_s, safety_s, guarantees) hold.
        """
        loop_positions = list(range(trace.lassoStart, trace.lengthOfTrace))

        if self.init_s is not None:
            if not _eval_prop(self.init_s, trace, 0):
                return False

        if self.safety_s is not None:
            for t in range(trace.lengthOfTrace):
                if not _eval_prop(self.safety_s, trace, t):
                    return False

        for g in self.guarantees:
            if not any(_eval_prop(g, trace, t) for t in loop_positions):
                return False

        return True

    def evaluate_on_trace(self, trace):
        """
        Evaluate this GR(1) formula on a lasso trace.
        Returns True if the trace satisfies the formula, False otherwise.

        Semantics:
        - Init: propositional formula must hold at position 0
        - Safety □φ: φ must hold at ALL positions
        - Justice □◇φ: φ must hold at SOME position in the loop
        - Full formula: assumptions → guarantees
        """
        loop_positions = list(range(trace.lassoStart, trace.lengthOfTrace))

        # Evaluate assumptions
        assumptions_hold = True

        if self.init_e is not None:
            if not _eval_prop(self.init_e, trace, 0):
                assumptions_hold = False

        if assumptions_hold and self.safety_e is not None:
            for t in range(trace.lengthOfTrace):
                if not _eval_prop(self.safety_e, trace, t):
                    assumptions_hold = False
                    break

        if assumptions_hold:
            for j in self.justices:
                if not any(_eval_prop(j, trace, t) for t in loop_positions):
                    assumptions_hold = False
                    break

        # If assumptions don't hold, implication is vacuously true
        if not assumptions_hold:
            return True

        # Evaluate guarantees
        if self.init_s is not None:
            if not _eval_prop(self.init_s, trace, 0):
                return False

        if self.safety_s is not None:
            for t in range(trace.lengthOfTrace):
                if not _eval_prop(self.safety_s, trace, t):
                    return False

        for g in self.guarantees:
            if not any(_eval_prop(g, trace, t) for t in loop_positions):
                return False

        return True

    def __repr__(self):
        return self.prettyPrint()


def _eval_prop(formula, trace, timestep):
    """
    Evaluate a propositional formula at a specific timestep of a trace.
    Handles &, |, !, ->, <->, propositional variables (x0, x1, ...),
    and primed variables (x0', x1', ...) for next-state.
    """
    label = formula.label

    if label.startswith('x') and label.endswith("'"):
        # Primed variable: value from successor position
        var_idx = int(label[1:-1])
        next_t = trace.nextPos(timestep)
        return trace.traceVector[next_t][var_idx]
    elif label.startswith('x'):
        var_idx = int(label[1:])
        return trace.traceVector[timestep][var_idx]
    elif label == '&':
        return _eval_prop(formula.left, trace, timestep) and _eval_prop(formula.right, trace, timestep)
    elif label == '|':
        return _eval_prop(formula.left, trace, timestep) or _eval_prop(formula.right, trace, timestep)
    elif label == '!':
        return not _eval_prop(formula.left, trace, timestep)
    elif label == '->':
        return (not _eval_prop(formula.left, trace, timestep)) or _eval_prop(formula.right, trace, timestep)
    elif label == '<->':
        return _eval_prop(formula.left, trace, timestep) == _eval_prop(formula.right, trace, timestep)
    elif label == 'true':
        return True
    elif label == 'false':
        return False
    else:
        raise ValueError("Unknown operator in propositional formula: " + label)
