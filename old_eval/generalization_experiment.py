"""
Generalization experiment: compare GR1Mine vs samples2LTL on held-out traces.

For each Spectra benchmark:
1. Generate a small training trace set from the ground truth spec
2. Mine with GR1Mine (incremental) → φ'  (GR(1) formula)
3. Mine with samples2LTL[SAT] → ψ        (flat LTL formula)
4. Generate a large held-out trace set from the ground truth
5. Classify held-out traces with:
   - φ  (ground truth GR(1): assumptions → guarantees)
   - φ' (mined GR(1): assumptions → guarantees)
   - ψ  (mined LTL, treated as pure guarantee: trace is positive iff ψ holds)
6. Report accuracy of φ' and ψ against φ's classifications
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
from utils.Traces import Trace, ExperimentTraces
from utils.spectra_parser import parse_spectra_file
from utils.trace_generator import generate_traces_sat
from experiments.eval_spectra import generate_traces_from_spec
from experiments.gr1_experiment import run_gr1_incremental_solver, verify_formula
from smtEncoding.dagSATEncoding import DagSATEncoding
from z3 import sat


SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

BENCHMARKS = [
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_2.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_trans_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_wo_ass_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_wo_ass_fairness_amba_ahb_2.spectra',
]


def mine_sat_baseline(traces, max_depth=10):
    """Mine an LTL formula using the SAT baseline."""
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    for D in range(1, max_depth + 1):
        encoder = DagSATEncoding(D, traces)
        encoder.encodeFormula()
        if encoder.solver.check() == sat:
            model = encoder.solver.model()
            return encoder.reconstructWholeFormula(model), D
    return None, None


def generate_heldout_traces(spec, num_vars, num_pos=100, num_neg=100):
    """Generate held-out traces using SAT-based construction.

    Uses Z3 to construct traces in the interesting region:
    - Genuine positives: assumptions AND guarantees hold
    - Assumption-satisfying negatives: assumptions hold, some guarantee fails
    - Vacuous positives: assumptions fail (random)

    This ensures the held-out set actually tests whether the mined formula
    captures the assume-guarantee structure correctly.
    """
    pos, neg = generate_traces_sat(
        spec, num_vars,
        num_pos=num_pos, num_neg=num_neg,
        trace_lengths=[4, 5, 6, 7],
        max_trials_per_config=max(10, num_neg)
    )
    return pos + neg


def classify_with_ltl(formula, trace):
    """Classify a trace using a flat LTL formula (pure guarantee: positive iff formula holds)."""
    return trace.evaluateFormulaOnTrace(formula)


def classify_with_gr1(formula, trace):
    """Classify a trace using a GR(1) formula (assumptions → guarantees)."""
    return formula.evaluate_on_trace(trace)


def run_experiment():
    print("=" * 130)
    print("Generalization Experiment: GR1Mine vs samples2LTL[SAT] on held-out traces")
    print("=" * 130)
    print()
    print("%-45s | %4s | %5s | %6s | %6s | %6s | %6s | %s" % (
        'Benchmark', 'Vars', 'Train', 'HO_P', 'HO_N',
        'GR1_acc', 'SAT_acc', 'SAT formula'))
    print("-" * 140)

    for bm in BENCHMARKS:
        path = os.path.join(SPECTRA_DIR, bm)
        if not os.path.exists(path):
            print("%-45s | SKIP — not found" % os.path.basename(bm))
            continue

        gr1_gt, meta = parse_spectra_file(path)
        n = meta['num_vars']
        nj = len(gr1_gt.justices)
        ng = len(gr1_gt.guarantees)
        name = os.path.basename(bm).replace('.spectra', '')

        # --- Training traces (SAT-constructed) ---
        pos_train, neg_train = generate_traces_sat(
            gr1_gt, n, num_pos=100, num_neg=100,
            trace_lengths=[4, 5, 6],
            max_trials_per_config=50
        )
        if len(neg_train) < 3:
            print("%-45s | %4d | SKIP — too few neg traces (%d)" % (name, n, len(neg_train)))
            continue
        num_train = len(pos_train) + len(neg_train)

        # --- Mine with GR1Mine ---
        traces_gr1 = ExperimentTraces(
            tracesToAccept=pos_train, tracesToReject=neg_train,
            operators=['&', '|', '!']
        )
        max_j = min(nj + 1, 4)
        max_g = min(ng + 1, 4)
        gr1_result = run_gr1_incremental_solver(traces_gr1, 5, max_j, max_g)
        gr1_mined = gr1_result[0]

        # --- Mine with SAT baseline ---
        traces_sat = ExperimentTraces(
            tracesToAccept=list(pos_train), tracesToReject=list(neg_train),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )
        sat_formula, sat_depth = mine_sat_baseline(traces_sat)

        if gr1_mined is None or sat_formula is None:
            print("%-45s | %4d | SKIP — mining failed (GR1=%s, SAT=%s)" % (
                name, n,
                'ok' if gr1_mined else 'FAIL',
                'ok' if sat_formula else 'FAIL'))
            continue

        # --- Held-out traces (SAT-constructed, assumption-satisfying) ---
        heldout = generate_heldout_traces(gr1_gt, n, num_pos=100, num_neg=100)
        num_ho_pos = sum(1 for t in heldout if t.intendedEvaluation)
        num_ho_neg = sum(1 for t in heldout if not t.intendedEvaluation)

        # --- Classify held-out with each mined formula ---
        gr1_correct = 0
        sat_correct = 0
        # Track errors by type
        gr1_fp = gr1_fn = sat_fp = sat_fn = 0

        for tr in heldout:
            ground_truth = tr.intendedEvaluation

            gr1_pred = classify_with_gr1(gr1_mined, tr)
            if gr1_pred == ground_truth:
                gr1_correct += 1
            elif gr1_pred and not ground_truth:
                gr1_fp += 1  # predicted positive, actually negative
            else:
                gr1_fn += 1  # predicted negative, actually positive

            sat_pred = classify_with_ltl(sat_formula, tr)
            if sat_pred == ground_truth:
                sat_correct += 1
            elif sat_pred and not ground_truth:
                sat_fp += 1
            else:
                sat_fn += 1

        gr1_acc = gr1_correct / len(heldout) * 100
        sat_acc = sat_correct / len(heldout) * 100

        sat_str = sat_formula.prettyPrint(True) if hasattr(sat_formula, 'prettyPrint') else str(sat_formula)
        if len(sat_str) > 30:
            sat_str = sat_str[:27] + '...'

        print("%-45s | %4d | %5d | %6d | %6d | %5.1f%% | %5.1f%% | %s" % (
            name, n, num_train, num_ho_pos, num_ho_neg,
            gr1_acc, sat_acc, sat_str))
        print("  └─ errors: GR1 FP=%d FN=%d | SAT FP=%d FN=%d" % (
            gr1_fp, gr1_fn, sat_fp, sat_fn))

    print()


if __name__ == '__main__':
    run_experiment()
