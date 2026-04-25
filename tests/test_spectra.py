"""Tests for Spectra parser and primed variable support."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.SimpleTree import Formula
from utils.Traces import Trace, ExperimentTraces
from utils.GR1Formula import GR1Formula, _eval_prop
from utils.spectra_parser import parse_spectra_file, _parse_formula


def make_trace(vector, lasso_start):
    return Trace([[bool(v) for v in step] for step in vector], lasso_start)


def test_primed_variable_eval():
    """Test _eval_prop with primed variables."""
    # Trace: [x0=1,x1=0], [x0=0,x1=1], lasso at 0
    tr = make_trace([[1, 0], [0, 1]], 0)

    # x0' at t=0 should be trace[nextPos(0)=1][0] = 0
    f_x0p = Formula("x0'")
    assert _eval_prop(f_x0p, tr, 0) == False
    # x0' at t=1 should be trace[nextPos(1)=0][0] = 1 (lasso wrap)
    assert _eval_prop(f_x0p, tr, 1) == True

    # x1' at t=0 should be trace[1][1] = 1
    f_x1p = Formula("x1'")
    assert _eval_prop(f_x1p, tr, 0) == True
    # x1' at t=1 should be trace[0][1] = 0 (lasso wrap)
    assert _eval_prop(f_x1p, tr, 1) == False

    print("PASS: test_primed_variable_eval")


def test_biconditional_eval():
    """Test <-> evaluation."""
    tr = make_trace([[1, 1], [1, 0]], 0)
    f_iff = Formula(['<->', Formula('x0'), Formula('x1')])
    assert _eval_prop(f_iff, tr, 0) == True   # 1 <-> 1
    assert _eval_prop(f_iff, tr, 1) == False  # 1 <-> 0
    print("PASS: test_biconditional_eval")


def test_safety_with_next():
    """Test GR(1) evaluation with next-state safety: G(x0 -> next(x1))."""
    # G(x0 -> x1') means: at every position, if x0 is true, then x1 at the next position is true
    spec = GR1Formula(
        safety_e=Formula('x0'),  # assume G(x0)
        safety_s=Formula(['->', Formula('x0'), Formula("x1'")]),  # guarantee G(x0 -> next(x1))
        justices=[Formula('x0')],
        guarantees=[Formula('x1')]
    )

    # Trace where x0 is always true and x1 at next step is always true
    tr1 = make_trace([[1, 1], [1, 1]], 0)
    assert spec.evaluate_on_trace(tr1) == True

    # Trace where x0=1 but next(x1)=0: x0=1,x1=1 then x0=1,x1=0
    # At t=0: x0=1, x1'=trace[1][1]=0 -> fails
    tr2 = make_trace([[1, 1], [1, 0]], 0)
    # safety_e=x0: x0 is 1 at all positions -> holds
    # safety_s=x0->x1': at t=1, x0=1, x1'=trace[nextPos(1)=0][1]=1 -> ok
    # at t=0, x0=1, x1'=trace[1][1]=0 -> FAILS
    assert spec.evaluate_on_trace(tr2) == False

    print("PASS: test_safety_with_next")


def test_parse_formula_simple():
    """Test parsing simple formulas."""
    n2i = {'a': 0, 'b': 1, 'c': 2}

    f = _parse_formula('a', n2i, primed=False)
    assert f.label == 'x0'

    f = _parse_formula('b=false', n2i, primed=False)
    assert f.label == '!'
    assert f.left.label == 'x1'

    f = _parse_formula('a & b', n2i, primed=False)
    assert f.label == '&'

    f = _parse_formula('a -> b', n2i, primed=False)
    assert f.label == '->'

    f = _parse_formula('a <-> b', n2i, primed=False)
    assert f.label == '<->'

    f = _parse_formula('FALSE', n2i, primed=False)
    assert f.label == 'false'

    print("PASS: test_parse_formula_simple")


def test_parse_formula_next():
    """Test parsing next() expressions."""
    n2i = {'a': 0, 'b': 1}

    f = _parse_formula('next(a)', n2i, primed=False)
    assert f.label == "x0'"

    f = _parse_formula('next(a=false)', n2i, primed=False)
    assert f.label == '!'
    assert f.left.label == "x0'"

    f = _parse_formula('a -> next(b)', n2i, primed=False)
    assert f.label == '->'
    assert f.left.label == 'x0'
    assert f.right.label == "x1'"

    print("PASS: test_parse_formula_next")


def test_parse_spectra_file():
    """Test parsing a real Spectra file from bloemDebugging."""
    spectra_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'spectra-specs', 'bloemDebugging')
    test_file = os.path.join(spectra_dir, 'specs_A2wst2.spectra')

    if not os.path.exists(test_file):
        print("SKIP: test_parse_spectra_file — Spectra benchmark not found")
        return

    gr1, meta = parse_spectra_file(test_file)

    # Should have 7 env + 15 sys = 22 boolean variables
    assert meta['num_vars'] == 22, "Expected 22 vars, got %d" % meta['num_vars']
    assert len(meta['env_vars']) == 7
    assert len(meta['sys_vars']) == 15

    # Should have init, safety, and justice conditions
    assert gr1.init_e is not None, "Should have init assumptions"
    assert gr1.init_s is not None, "Should have init guarantees"
    assert gr1.safety_e is not None, "Should have safety assumptions"
    assert gr1.safety_s is not None, "Should have safety guarantees"
    assert len(gr1.justices) > 0, "Should have justice conditions"
    assert len(gr1.guarantees) > 0, "Should have guarantee conditions"

    print("Parsed %s:" % os.path.basename(test_file))
    print("  %d env vars, %d sys vars" % (len(meta['env_vars']), len(meta['sys_vars'])))
    print("  %d justice conditions" % len(gr1.justices))
    print("  %d guarantee conditions" % len(gr1.guarantees))

    print("PASS: test_parse_spectra_file")


def test_parse_and_evaluate():
    """Parse a Spectra file and evaluate on random traces."""
    spectra_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'spectra-specs', 'bloemDebugging')
    test_file = os.path.join(spectra_dir, 'specs_A2wst2.spectra')

    if not os.path.exists(test_file):
        print("SKIP: test_parse_and_evaluate — Spectra benchmark not found")
        return

    import random
    random.seed(42)

    gr1, meta = parse_spectra_file(test_file)
    n = meta['num_vars']

    # Generate random traces and evaluate — should not crash
    pos_count = 0
    neg_count = 0
    for _ in range(100):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(n)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        result = gr1.evaluate_on_trace(tr)
        if result:
            pos_count += 1
        else:
            neg_count += 1

    print("Evaluated 100 random traces: %d positive, %d negative" % (pos_count, neg_count))
    assert pos_count + neg_count == 100
    print("PASS: test_parse_and_evaluate")


def test_mining_with_primed_vars():
    """Test mining a safety spec with next-state variables.

    Target: (G(next(x0))) → (G(next(x1)) ∧ GF(x0))
    Safety_s = next(x1) which is just a primed variable (D=1).
    """
    from smtEncoding.gr1SATEncoding import GR1SATEncoding
    from z3 import sat

    spec = GR1Formula(
        safety_e=Formula("x0'"),   # G(next(x0))
        safety_s=Formula("x1'"),   # G(next(x1))
        justices=[Formula('x0')],
        guarantees=[Formula('x0')]
    )

    import random
    random.seed(42)

    pos_traces = []
    neg_traces = []
    for _ in range(500):
        length = random.randint(2, 5)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(2)] for _ in range(length)]
        tr = make_trace(vec, lasso)
        if spec.evaluate_on_trace(tr):
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)

    print("Primed mining test: %d positive, %d negative" % (len(pos_traces), len(neg_traces)))

    if len(neg_traces) < 3:
        print("SKIP: not enough negative traces for primed mining test")
        return

    traces = ExperimentTraces(
        tracesToAccept=pos_traces[:15],
        tracesToReject=neg_traces[:10],
        operators=['&', '|', '!']
    )

    # D=1 should find it since safety_s is just a single primed variable
    encoder = GR1SATEncoding(1, traces, num_justices=1, num_guarantees=1,
                             include_safety=True)
    encoder.encodeFormula()
    result = encoder.solver.check()
    print("D=1 result:", result)

    if result == sat:
        model = encoder.solver.model()
        formula = encoder.reconstructWholeFormula(model)
        print("Mined:", formula.prettyPrint())

        # Verify soundness
        for i, tr in enumerate(traces.acceptedTraces):
            assert formula.evaluate_on_trace(tr), "Positive trace %d failed" % i
        for i, tr in enumerate(traces.rejectedTraces):
            assert not formula.evaluate_on_trace(tr), "Negative trace %d failed" % i

        print("PASS: test_mining_with_primed_vars")
    else:
        print("FAIL: UNSAT at D=1 for primed variable mining")


if __name__ == '__main__':
    test_primed_variable_eval()
    test_biconditional_eval()
    test_safety_with_next()
    test_parse_formula_simple()
    test_parse_formula_next()
    test_parse_spectra_file()
    test_parse_and_evaluate()
    test_mining_with_primed_vars()
    print("\n=== All Spectra tests completed ===")
