"""
Test suite for GR(1) specification mining.

Tests the GR1Formula evaluation and GR1SATEncoding on hand-crafted traces.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.SimpleTree import Formula
from utils.Traces import Trace, ExperimentTraces
from utils.GR1Formula import GR1Formula, _eval_prop
from smtEncoding.gr1SATEncoding import GR1SATEncoding


def make_trace(vector, lasso_start):
    """Helper to create a Trace from a list of lists."""
    return Trace([[bool(v) for v in step] for step in vector], lasso_start)


def test_eval_prop():
    """Test propositional formula evaluation on a trace."""
    # Trace: x0=1,x1=0 at time 0; x0=0,x1=1 at time 1
    tr = make_trace([[1, 0], [0, 1]], 0)

    f_x0 = Formula('x0')
    f_x1 = Formula('x1')
    f_and = Formula(['&', Formula('x0'), Formula('x1')])
    f_or = Formula(['|', Formula('x0'), Formula('x1')])
    f_not = Formula(['!', Formula('x0')])

    assert _eval_prop(f_x0, tr, 0) == True
    assert _eval_prop(f_x0, tr, 1) == False
    assert _eval_prop(f_x1, tr, 0) == False
    assert _eval_prop(f_x1, tr, 1) == True
    assert _eval_prop(f_and, tr, 0) == False  # 1 & 0
    assert _eval_prop(f_or, tr, 0) == True   # 1 | 0
    assert _eval_prop(f_not, tr, 0) == False  # !1
    assert _eval_prop(f_not, tr, 1) == True   # !0
    print("PASS: test_eval_prop")


def test_gr1_evaluate_simple():
    """Test GR(1) formula evaluation: □◇x0 → □◇x1"""
    # Justice: x0, Guarantee: x1
    spec = GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')])

    # Positive: x0 and x1 both appear in loop → assumptions hold, guarantees hold
    tr1 = make_trace([[1, 1], [0, 0], [1, 1]], 1)  # loop: [0,0],[1,1] — x0 and x1 in loop
    assert spec.evaluate_on_trace(tr1) == True, "Both hold in loop"

    # Positive (vacuous): x0 never in loop → assumptions fail → vacuously true
    tr2 = make_trace([[1, 0], [0, 0], [0, 0]], 1)  # loop: [0,0],[0,0] — x0 never in loop
    assert spec.evaluate_on_trace(tr2) == True, "Vacuously true"

    # Negative: x0 in loop but x1 never in loop
    tr3 = make_trace([[1, 0], [1, 0], [1, 0]], 0)  # loop: all [1,0] — x0 yes, x1 no
    assert spec.evaluate_on_trace(tr3) == False, "Assumption holds but guarantee fails"

    print("PASS: test_gr1_evaluate_simple")


def test_gr1_evaluate_negation():
    """Test GR(1) formula evaluation: □◇(¬x0) → □◇x1"""
    spec = GR1Formula(justices=[Formula(['!', Formula('x0')])],
                      guarantees=[Formula('x1')])

    # Positive: ¬x0 in loop and x1 in loop
    tr1 = make_trace([[0, 1], [1, 0], [0, 1]], 1)  # loop has x0=1 and x0=0, x1=0 and x1=1
    assert spec.evaluate_on_trace(tr1) == True

    # Negative: ¬x0 in loop (x0=0 somewhere) but x1 never in loop
    tr2 = make_trace([[0, 0], [0, 0]], 0)  # all zeros — ¬x0 holds, x1 never
    assert spec.evaluate_on_trace(tr2) == False

    print("PASS: test_gr1_evaluate_negation")


def test_gr1_mining_simple():
    """
    Test mining □◇J → □◇G with 2 variables.
    Target: □◇x0 → □◇x1
    """
    # Positive traces: either x0 not in loop (vacuous), or both x0 and x1 in loop
    pos1 = make_trace([[1, 1], [1, 1]], 0)    # x0 and x1 always → both in loop
    pos2 = make_trace([[0, 0], [0, 1]], 0)    # x0 never in loop → vacuous
    pos3 = make_trace([[1, 1], [0, 1]], 0)    # x0 in loop, x1 in loop
    pos4 = make_trace([[0, 0], [0, 0]], 0)    # x0 never → vacuous
    pos5 = make_trace([[1, 0], [0, 1], [1, 1]], 1)  # loop: [0,1],[1,1] — both present

    # Negative traces: x0 in loop but x1 NOT in loop
    neg1 = make_trace([[1, 0], [1, 0]], 0)    # x0 always, x1 never
    neg2 = make_trace([[1, 0], [1, 0], [1, 0]], 0)  # same
    neg3 = make_trace([[0, 0], [1, 0], [1, 0]], 1)  # loop: [1,0],[1,0] — x0 yes, x1 no

    traces = ExperimentTraces(
        tracesToAccept=[pos1, pos2, pos3, pos4, pos5],
        tracesToReject=[neg1, neg2, neg3],
        operators=['&', '|', '!'],
        depth=1
    )

    # Try D=1 (each component is a single variable)
    encoder = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1)
    encoder.encodeFormula()

    from z3 import sat
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined GR(1) spec:", formula.prettyPrint())

        # Verify the mined spec is consistent with all traces
        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr) == True, \
                "Mined formula should accept positive trace %d" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert formula.evaluate_on_trace(tr) == False, \
                "Mined formula should reject negative trace %d" % i

        print("PASS: test_gr1_mining_simple — mined spec is consistent")
    else:
        print("FAIL: test_gr1_mining_simple — UNSAT at D=1")


def test_gr1_mining_negation():
    """
    Test mining □◇J → □◇G with negation needed.
    Target: □◇(¬x0) → □◇x1
    With literal-leaves encoding, D=1 can express ¬x0 (single negated leaf).
    """
    spec = GR1Formula(justices=[Formula(['!', Formula('x0')])],
                      guarantees=[Formula('x1')])

    # Generate traces by evaluating the target spec
    pos_traces = []
    neg_traces = []

    import random
    random.seed(42)
    for _ in range(200):
        length = random.randint(2, 4)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(2)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Generated %d positive, %d negative traces" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 3:
        print("SKIP: not enough negative traces generated")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:10],
        tracesToReject=neg_traces[:5],
        operators=['&', '|', '!'],
        depth=1
    )

    # D=1 should succeed (leaf polarity handles negation)
    enc1 = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1)
    enc1.encodeFormula()
    from z3 import sat
    r1 = enc1.solver.check()
    print("D=1 result:", r1)

    if r1 == sat:
        model = enc1.solver.model()
        formula = enc1.reconstructWholeFormula(model)
        print("Mined GR(1) spec:", formula.prettyPrint())

        # Verify consistency
        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr) == True, \
                "Positive trace %d should be accepted" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert formula.evaluate_on_trace(tr) == False, \
                "Negative trace %d should be rejected" % i

        print("PASS: test_gr1_mining_negation")
    else:
        print("FAIL: test_gr1_mining_negation — UNSAT at D=1")


def test_gr1_evaluate_full():
    """Test full GR(1) evaluation with init + safety + justice."""
    # (x0 ∧ □x1 ∧ □◇x2) → (x3 ∧ □x0 ∧ □◇x1)
    # 4 variables: x0, x1, x2, x3
    spec = GR1Formula(
        init_e=Formula('x0'),
        safety_e=Formula('x1'),
        justices=[Formula('x2')],
        init_s=Formula('x3'),
        safety_s=Formula('x0'),
        guarantees=[Formula('x1')]
    )

    # Positive: all assumptions and guarantees hold
    # x0=1,x1=1,x2=1,x3=1 everywhere
    tr1 = make_trace([[1, 1, 1, 1], [1, 1, 1, 1]], 0)
    assert spec.evaluate_on_trace(tr1) == True, "All hold"

    # Positive (vacuous): init_e fails (x0=0 at position 0)
    tr2 = make_trace([[0, 1, 1, 0], [0, 1, 1, 0]], 0)
    assert spec.evaluate_on_trace(tr2) == True, "Vacuous — init_e fails"

    # Positive (vacuous): safety_e fails (x1=0 at some position)
    tr3 = make_trace([[1, 1, 1, 1], [1, 0, 1, 1]], 0)
    assert spec.evaluate_on_trace(tr3) == True, "Vacuous — safety_e fails"

    # Negative: all assumptions hold but init_s fails (x3=0 at position 0)
    tr4 = make_trace([[1, 1, 1, 0], [1, 1, 1, 1]], 0)
    assert spec.evaluate_on_trace(tr4) == False, "init_s fails"

    # Negative: all assumptions hold but safety_s fails (x0=0 at some position)
    tr5 = make_trace([[1, 1, 1, 1], [0, 1, 1, 1]], 0)
    # safety_e checks x1: x1=1,1 ✓. safety_s checks x0: x0=1,0 ✗
    # But wait — safety_e checks x1 at ALL positions, and x1=1 at both. ✓
    # assumptions: init_e(x0=1)✓, safety_e(x1 always)✓, justice(x2 in loop)✓
    # guarantees: init_s(x3=1)✓, safety_s(x0 always) — x0=0 at pos 1 ✗
    assert spec.evaluate_on_trace(tr5) == False, "safety_s fails"

    print("PASS: test_gr1_evaluate_full")


def test_gr1_mining_with_init():
    """Test mining with init conditions: (x0 ∧ □◇x1) → (x2 ∧ □◇x0)"""
    import random
    random.seed(123)

    spec = GR1Formula(
        init_e=Formula('x0'),
        justices=[Formula('x1')],
        init_s=Formula('x2'),
        guarantees=[Formula('x0')]
    )

    pos_traces = []
    neg_traces = []
    for _ in range(500):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(3)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Init test: %d positive, %d negative" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 5:
        print("SKIP: not enough negative traces")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    from z3 import sat
    encoder = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1,
                             include_init=True)
    encoder.encodeFormula()
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined:", formula.prettyPrint())

        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr), "Positive trace %d failed" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert not formula.evaluate_on_trace(tr), "Negative trace %d failed" % i
        print("PASS: test_gr1_mining_with_init")
    else:
        print("FAIL: UNSAT at D=1")


def test_gr1_mining_with_safety():
    """Test mining with safety: (□x0 ∧ □◇x1) → (□x2 ∧ □◇x0)"""
    import random
    random.seed(456)

    spec = GR1Formula(
        safety_e=Formula('x0'),
        justices=[Formula('x1')],
        safety_s=Formula('x2'),
        guarantees=[Formula('x0')]
    )

    pos_traces = []
    neg_traces = []
    for _ in range(500):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(3)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Safety test: %d positive, %d negative" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 5:
        print("SKIP: not enough negative traces")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    from z3 import sat
    encoder = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1,
                             include_safety=True)
    encoder.encodeFormula()
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined:", formula.prettyPrint())

        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr), "Positive trace %d failed" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert not formula.evaluate_on_trace(tr), "Negative trace %d failed" % i
        print("PASS: test_gr1_mining_with_safety")
    else:
        print("FAIL: UNSAT at D=1")


def test_gr1_mining_multiple_justices():
    """Test mining with 2 justice conditions: (□◇x0 ∧ □◇x1) → □◇x2"""
    import random
    random.seed(789)

    spec = GR1Formula(
        justices=[Formula('x0'), Formula('x1')],
        guarantees=[Formula('x2')]
    )

    pos_traces = []
    neg_traces = []
    for _ in range(500):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(3)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Multi-justice test: %d positive, %d negative" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 5:
        print("SKIP: not enough negative traces")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    from z3 import sat
    encoder = GR1SATEncoding(1, traces, num_justices=2, num_guarantees=1)
    encoder.encodeFormula()
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined:", formula.prettyPrint())

        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr), "Positive trace %d failed" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert not formula.evaluate_on_trace(tr), "Negative trace %d failed" % i
        print("PASS: test_gr1_mining_multiple_justices")
    else:
        print("FAIL: UNSAT at D=1")


def test_gr1_mining_full():
    """Test full GR(1): (x0 ∧ □x1 ∧ □◇x2) → (x3 ∧ □x0 ∧ □◇x1)"""
    import random
    random.seed(999)

    spec = GR1Formula(
        init_e=Formula('x0'),
        safety_e=Formula('x1'),
        justices=[Formula('x2')],
        init_s=Formula('x3'),
        safety_s=Formula('x0'),
        guarantees=[Formula('x1')]
    )

    pos_traces = []
    neg_traces = []
    for _ in range(1000):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(4)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Full GR(1) test: %d positive, %d negative" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 5:
        print("SKIP: not enough negative traces")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    from z3 import sat
    encoder = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1,
                             include_init=True, include_safety=True)
    encoder.encodeFormula()
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined:", formula.prettyPrint())

        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr), "Positive trace %d failed" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert not formula.evaluate_on_trace(tr), "Negative trace %d failed" % i
        print("PASS: test_gr1_mining_full")
    else:
        print("FAIL: UNSAT at D=1")


def test_enumerating_solver_simple():
    """Test that the enumerating solver discovers the right template for □◇x0 → □◇x1."""
    from experiments.gr1_experiment import run_gr1_enumerating_solver

    pos1 = make_trace([[1, 1], [1, 1]], 0)
    pos2 = make_trace([[0, 0], [0, 1]], 0)
    pos3 = make_trace([[1, 1], [0, 1]], 0)
    pos4 = make_trace([[0, 0], [0, 0]], 0)

    neg1 = make_trace([[1, 0], [1, 0]], 0)
    neg2 = make_trace([[1, 0], [1, 0], [1, 0]], 0)
    neg3 = make_trace([[0, 0], [1, 0], [1, 0]], 1)

    traces = ExperimentTraces(
        tracesToAccept=[pos1, pos2, pos3, pos4],
        tracesToReject=[neg1, neg2, neg3],
        operators=['&', '|', '!'],
        depth=1
    )

    result = run_gr1_enumerating_solver(traces, max_depth=3, max_justices=2, max_guarantees=2)
    formula, D, elapsed, config = result

    assert formula is not None, "Should find a solution"
    assert D == 1, "Should find at D=1"
    inc_init, inc_safety, m, k = config
    assert not inc_init, "Should not need init"
    assert not inc_safety, "Should not need safety"
    assert m == 1, "Should find 1 justice"
    assert k == 1, "Should find 1 guarantee"

    # Verify soundness
    for tr in traces.acceptedTraces:
        assert formula.evaluate_on_trace(tr), "Should accept positive traces"
    for tr in traces.rejectedTraces:
        assert not formula.evaluate_on_trace(tr), "Should reject negative traces"

    print("PASS: test_enumerating_solver_simple — found template J with m=1,k=1")


def test_enumerating_solver_multi_justice():
    """Test that the enumerating solver discovers 2 justice conditions."""
    import random
    random.seed(789)
    from experiments.gr1_experiment import run_gr1_enumerating_solver

    spec = GR1Formula(
        justices=[Formula('x0'), Formula('x1')],
        guarantees=[Formula('x2')]
    )

    pos_traces = []
    neg_traces = []
    for _ in range(500):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(3)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    result = run_gr1_enumerating_solver(traces, max_depth=3, max_justices=3, max_guarantees=3)
    formula, D, elapsed, config = result

    assert formula is not None, "Should find a solution"
    # Verify soundness
    for tr in traces.acceptedTraces:
        assert formula.evaluate_on_trace(tr), "Should accept positive traces"
    for tr in traces.rejectedTraces:
        assert not formula.evaluate_on_trace(tr), "Should reject negative traces"

    print("PASS: test_enumerating_solver_multi_justice — config=%s, D=%d" % (str(config), D))


if __name__ == '__main__':
    test_eval_prop()
    test_gr1_evaluate_simple()
    test_gr1_evaluate_negation()
    test_gr1_evaluate_full()
    test_gr1_mining_simple()
    test_gr1_mining_negation()
    test_gr1_mining_with_init()
    test_gr1_mining_with_safety()
    test_gr1_mining_multiple_justices()
    test_gr1_mining_full()
    test_enumerating_solver_simple()
    test_enumerating_solver_multi_justice()
    print("\n=== All tests completed ===")
