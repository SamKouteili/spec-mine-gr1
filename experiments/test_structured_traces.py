"""
Quick comparison: random vs SAT-based trace generation.
Runs GR1Mine + SAT + DT on small Spectra benchmarks with both strategies,
printing the mined formulas to show whether baselines produce spurious specs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import time
from multiprocessing import Process, Queue

from utils.Traces import Trace, ExperimentTraces
from utils.spectra_parser import parse_spectra_file
from utils.trace_generator import generate_traces_sat
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from smtEncoding.dagSATEncoding import DagSATEncoding
from solverRuns import run_dt_solver


TIMEOUT = 300  # 5 minutes


# ============================================================
# Old random trace generation (baseline, for comparison)
# ============================================================

def generate_traces_random(spec, num_vars, num_pos=15, num_neg=10,
                           trace_length=5, max_trials=20000, seed=42):
    """Old-style: purely random, no structure awareness."""
    random.seed(seed)
    pos, neg = [], []
    for _ in range(max_trials):
        length = random.randint(max(2, trace_length - 1), trace_length + 1)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(num_vars)] for _ in range(length)]
        tr = Trace([[bool(v) for v in step] for step in vec], lasso)
        if spec.evaluate_on_trace(tr):
            tr.intendedEvaluation = True
            pos.append(tr)
        else:
            tr.intendedEvaluation = False
            neg.append(tr)
        if len(pos) >= num_pos * 3 and len(neg) >= num_neg * 3:
            break
    return pos[:num_pos], neg[:num_neg]


# ============================================================
# SAT-based trace generation wrapper
# ============================================================

def generate_traces_sat_wrapper(spec, num_vars, num_pos=15, num_neg=10,
                                trace_length=5, max_trials=50000, seed=42):
    """Wrapper around SAT-based generator with trace breakdown reporting."""
    pos, neg = generate_traces_sat(
        spec, num_vars, num_pos=num_pos, num_neg=num_neg,
        trace_lengths=[trace_length - 1, trace_length, trace_length + 1])

    n_neg_assum = sum(1 for t in neg if spec.evaluate_assumptions(t))
    n_pos_genuine = sum(1 for t in pos if spec.evaluate_assumptions(t))
    print("    Trace breakdown: neg(assum_hold=%d, other=%d), "
          "pos(genuine=%d, vacuous=%d)" % (
              n_neg_assum, len(neg) - n_neg_assum,
              n_pos_genuine, len(pos) - n_pos_genuine))
    return pos, neg


# ============================================================
# Workers
# ============================================================

def _run_gr1_worker(q, traces, max_depth, max_j, max_g):
    from experiments.gr1_experiment import generate_template_configs, verify_formula
    from z3 import sat
    start = time.time()
    configs = generate_template_configs(max_j, max_g)
    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            enc = GR1SATEncoding(D, traces, num_justices=m, num_guarantees=k,
                                 include_init=include_init, include_safety=include_safety)
            enc.encodeFormula()
            if enc.solver.check() == sat:
                model = enc.solver.model()
                formula = enc.reconstructWholeFormula(model)
                if verify_formula(formula, traces):
                    q.put(('gr1', str(formula), D, time.time() - start))
                    return
    q.put(('gr1', None, None, time.time() - start))


def _run_sat_worker(q, traces, max_depth):
    from z3 import sat
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    start = time.time()
    for D in range(1, max_depth + 1):
        enc = DagSATEncoding(D, traces)
        enc.encodeFormula()
        if enc.solver.check() == sat:
            model = enc.solver.model()
            formula = enc.reconstructWholeFormula(model)
            q.put(('sat', formula.prettyPrint(True), D, time.time() - start))
            return
    q.put(('sat', None, None, time.time() - start))


def _run_dt_worker(q, traces):
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    try:
        result = run_dt_solver(traces, encoder=DagSATEncoding)
        q.put(('dt', str(result[0]), result[1], result[2]))
    except Exception:
        q.put(('dt', None, None, None))


def run_with_timeout(worker_fn, args, timeout=TIMEOUT):
    q = Queue()
    p = Process(target=worker_fn, args=(q, *args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not q.empty():
        return q.get()
    return None


# ============================================================
# Main
# ============================================================

def main():
    spectra_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

    # Pick small benchmarks (16 vars)
    test_specs = [
        os.path.join(spectra_dir, 'cimattiAnalyzing', 'amba_ahb_wo_ass_fairness_amba_ahb_1.spectra'),
        os.path.join(spectra_dir, 'cimattiAnalyzing', 'amba_ahb_w_guar_trans_amba_ahb_1.spectra'),
    ]

    for spec_path in test_specs:
        name = os.path.basename(spec_path).replace('.spectra', '')
        try:
            gr1, meta = parse_spectra_file(spec_path)
        except Exception as e:
            print("SKIP %s: %s" % (name, e))
            continue

        num_vars = meta['num_vars']
        nj = len(gr1.justices)
        ng = len(gr1.guarantees)
        print("=" * 80)
        print("BENCHMARK: %s  (vars=%d, J=%d, G=%d)" % (name, num_vars, nj, ng))
        print("Ground truth: %s" % gr1.prettyPrint())
        print("=" * 80)

        configs = [
            ("RANDOM_15+10_len5", generate_traces_random, 15, 10, 5),
            ("SAT_50+30_len5", generate_traces_sat_wrapper, 50, 30, 5),
            ("SAT_100+60_len8", generate_traces_sat_wrapper, 100, 60, 8),
        ]
        for gen_name, gen_fn, np, nn, tl in configs:
            print("\n  --- %s ---" % gen_name)
            pos, neg = gen_fn(gr1, num_vars, num_pos=np, num_neg=nn,
                              trace_length=tl, max_trials=50000, seed=42)
            print("    Generated: %d pos, %d neg" % (len(pos), len(neg)))

            if len(neg) < 3:
                print("    SKIP — insufficient negative traces")
                continue

            # Run all three miners
            traces_gr1 = ExperimentTraces(
                tracesToAccept=pos, tracesToReject=neg, operators=['&', '|', '!'])

            # GR1Mine
            gr1_result = run_with_timeout(
                _run_gr1_worker,
                (traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4)),
                timeout=TIMEOUT)
            if gr1_result and gr1_result[1]:
                print("    GR1Mine:  %.3fs  D=%s  formula=%s" % (
                    gr1_result[3], gr1_result[2], gr1_result[1]))
            else:
                print("    GR1Mine:  TIMEOUT or UNSAT")

            # SAT baseline
            traces_sat = ExperimentTraces(
                tracesToAccept=list(pos), tracesToReject=list(neg),
                operators=['G', 'F', '!', 'U', '&', '|', '->', 'X'])
            sat_result = run_with_timeout(
                _run_sat_worker, (traces_sat, 10), timeout=TIMEOUT)
            if sat_result and sat_result[1]:
                print("    SAT:      %.3fs  D=%s  formula=%s" % (
                    sat_result[3], sat_result[2], sat_result[1]))
            else:
                print("    SAT:      TIMEOUT or UNSAT")

            # DT baseline
            traces_dt = ExperimentTraces(
                tracesToAccept=list(pos), tracesToReject=list(neg),
                operators=['G', 'F', '!', 'U', '&', '|', '->', 'X'])
            dt_result = run_with_timeout(
                _run_dt_worker, (traces_dt,), timeout=TIMEOUT)
            if dt_result and dt_result[1]:
                print("    DT:       %ss  atoms=%s  prims=%s" % (
                    dt_result[1], dt_result[2], dt_result[3]))
            else:
                print("    DT:       TIMEOUT or ERROR")

        print()


if __name__ == '__main__':
    main()
