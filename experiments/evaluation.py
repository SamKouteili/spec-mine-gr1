"""
Full evaluation: GR(1) miner vs baseline Neider-Gavran LTL miner.

Generates traces from known GR(1) specs, runs both miners with a 10-minute timeout,
and reports time, formula size, and correctness.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import signal
import csv
import time
from multiprocessing import Process, Queue

from utils.SimpleTree import Formula
from utils.Traces import Trace, ExperimentTraces
from utils.GR1Formula import GR1Formula
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from smtEncoding.dagSATEncoding import DagSATEncoding
from solverRuns import run_dt_solver


TIMEOUT = 300  # 5 minutes


# ============================================================
# Benchmark definitions
# ============================================================

def make_benchmarks():
    """Define benchmark GR(1) specs with metadata."""
    benchmarks = []

    # --- Justice-only benchmarks ---

    # B1: □◇x0 → □◇x1 (simplest, D=1, 2 vars)
    benchmarks.append({
        'name': 'B01_simple_2v',
        'spec': GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')]),
        'num_vars': 2, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B2: □◇x0 → □◇x1 with 3 vars (superfluous x2)
    benchmarks.append({
        'name': 'B02_simple_3v',
        'spec': GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B3: □◇x0 → □◇x1 with 4 vars
    benchmarks.append({
        'name': 'B03_simple_4v',
        'spec': GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')]),
        'num_vars': 4, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B4: □◇x0 → □◇x1 with 5 vars
    benchmarks.append({
        'name': 'B04_simple_5v',
        'spec': GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')]),
        'num_vars': 5, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B5: □◇(¬x0) → □◇x1 (D=2, negation)
    benchmarks.append({
        'name': 'B05_neg_2v',
        'spec': GR1Formula(justices=[Formula(['!', Formula('x0')])],
                           guarantees=[Formula('x1')]),
        'num_vars': 2, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 2,
    })

    # B6: □◇(¬x0) → □◇x1 with 4 vars
    benchmarks.append({
        'name': 'B06_neg_4v',
        'spec': GR1Formula(justices=[Formula(['!', Formula('x0')])],
                           guarantees=[Formula('x1')]),
        'num_vars': 4, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 2,
    })

    # B7: □◇(x0 ∧ x1) → □◇x2 (D=2, conjunction in justice)
    benchmarks.append({
        'name': 'B07_conj_3v',
        'spec': GR1Formula(
            justices=[Formula(['&', Formula('x0'), Formula('x1')])],
            guarantees=[Formula('x2')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 2,
    })

    # B8: □◇(x0 ∧ x1) → □◇(x2 | x3) (D=2 both sides, 4 vars)
    benchmarks.append({
        'name': 'B08_conj_disj_4v',
        'spec': GR1Formula(
            justices=[Formula(['&', Formula('x0'), Formula('x1')])],
            guarantees=[Formula(['|', Formula('x2'), Formula('x3')])]),
        'num_vars': 4, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 2,
    })

    # B9: □◇(x0 ∧ ¬x1) → □◇x2 (D=3, nested)
    benchmarks.append({
        'name': 'B09_nested_3v',
        'spec': GR1Formula(
            justices=[Formula(['&', Formula('x0'), Formula(['!', Formula('x1')])])],
            guarantees=[Formula('x2')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 3,
    })

    # --- Multiple justice/guarantee ---

    # B10: (□◇x0 ∧ □◇x1) → □◇x2 (2 justices)
    benchmarks.append({
        'name': 'B10_2just_3v',
        'spec': GR1Formula(
            justices=[Formula('x0'), Formula('x1')],
            guarantees=[Formula('x2')]),
        'num_vars': 3, 'num_justices': 2, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B11: □◇x0 → (□◇x1 ∧ □◇x2) (2 guarantees)
    benchmarks.append({
        'name': 'B11_2guar_3v',
        'spec': GR1Formula(
            justices=[Formula('x0')],
            guarantees=[Formula('x1'), Formula('x2')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 2,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # B12: (□◇x0 ∧ □◇x1) → (□◇x2 ∧ □◇x3) (2+2, 4 vars)
    benchmarks.append({
        'name': 'B12_2x2_4v',
        'spec': GR1Formula(
            justices=[Formula('x0'), Formula('x1')],
            guarantees=[Formula('x2'), Formula('x3')]),
        'num_vars': 4, 'num_justices': 2, 'num_guarantees': 2,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    # --- Init benchmarks ---

    # B13: (x0 ∧ □◇x1) → (x2 ∧ □◇x0)
    benchmarks.append({
        'name': 'B13_init_3v',
        'spec': GR1Formula(
            init_e=Formula('x0'), justices=[Formula('x1')],
            init_s=Formula('x2'), guarantees=[Formula('x0')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': True, 'include_safety': False,
        'expected_D': 1,
    })

    # --- Safety benchmarks ---

    # B14: (□x0 ∧ □◇x1) → (□x2 ∧ □◇x0)
    benchmarks.append({
        'name': 'B14_safety_3v',
        'spec': GR1Formula(
            safety_e=Formula('x0'), justices=[Formula('x1')],
            safety_s=Formula('x2'), guarantees=[Formula('x0')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': True,
        'expected_D': 1,
    })

    # --- Full GR(1) benchmarks ---

    # B15: (x0 ∧ □x1 ∧ □◇x2) → (x3 ∧ □x0 ∧ □◇x1)
    benchmarks.append({
        'name': 'B15_full_4v',
        'spec': GR1Formula(
            init_e=Formula('x0'), safety_e=Formula('x1'), justices=[Formula('x2')],
            init_s=Formula('x3'), safety_s=Formula('x0'), guarantees=[Formula('x1')]),
        'num_vars': 4, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': True, 'include_safety': True,
        'expected_D': 1,
    })

    # B16: full with 5 vars
    benchmarks.append({
        'name': 'B16_full_5v',
        'spec': GR1Formula(
            init_e=Formula('x0'), safety_e=Formula('x1'), justices=[Formula('x2')],
            init_s=Formula('x3'), safety_s=Formula('x4'), guarantees=[Formula('x1')]),
        'num_vars': 5, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': True, 'include_safety': True,
        'expected_D': 1,
    })

    # B17: full with 2 justices + 2 guarantees, 4 vars
    benchmarks.append({
        'name': 'B17_full_2x2_4v',
        'spec': GR1Formula(
            init_e=Formula('x0'), safety_e=Formula('x1'),
            justices=[Formula('x2'), Formula('x3')],
            init_s=Formula('x1'), safety_s=Formula('x0'),
            guarantees=[Formula('x3'), Formula('x2')]),
        'num_vars': 4, 'num_justices': 2, 'num_guarantees': 2,
        'include_init': True, 'include_safety': True,
        'expected_D': 1,
    })

    # B18: full with D=2 justice (negation), 3 vars
    benchmarks.append({
        'name': 'B18_full_neg_3v',
        'spec': GR1Formula(
            init_e=Formula('x0'), safety_e=Formula('x1'),
            justices=[Formula(['!', Formula('x2')])],
            init_s=Formula('x1'), safety_s=Formula('x0'),
            guarantees=[Formula('x2')]),
        'num_vars': 3, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': True, 'include_safety': True,
        'expected_D': 2,
    })

    # B19: (□◇(x0|x1) ∧ □◇x2) → □◇(x1 ∧ x2) with 4 vars
    benchmarks.append({
        'name': 'B19_complex_4v',
        'spec': GR1Formula(
            justices=[Formula(['|', Formula('x0'), Formula('x1')]),
                      Formula('x2')],
            guarantees=[Formula(['&', Formula('x1'), Formula('x2')])]),
        'num_vars': 4, 'num_justices': 2, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 2,
    })

    # B20: scaling test — justice-only, 6 vars, D=1
    benchmarks.append({
        'name': 'B20_scale_6v',
        'spec': GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')]),
        'num_vars': 6, 'num_justices': 1, 'num_guarantees': 1,
        'include_init': False, 'include_safety': False,
        'expected_D': 1,
    })

    return benchmarks


# ============================================================
# Trace generation
# ============================================================

def generate_traces(spec, num_vars, num_pos=10, num_neg=10, trace_length=5, seed=42):
    """
    Generate structured traces that prevent spurious separators.

    Prioritizes assumption-satisfying negatives (forces any correct separator
    to get the guarantee side right) and includes both genuine and vacuous
    positives (prevents separators that are too specific on the assumption side).
    """
    random.seed(seed)

    neg_assum_hold = []
    neg_assum_fail = []
    pos_genuine = []
    pos_vacuous = []

    for _ in range(10000):
        length = random.randint(max(2, trace_length - 1), trace_length + 1)
        lasso = random.randint(0, length - 1)
        vec = [[random.randint(0, 1) for _ in range(num_vars)] for _ in range(length)]
        tr = Trace([[bool(v) for v in step] for step in vec], lasso)

        result = spec.evaluate_on_trace(tr)
        assum = spec.evaluate_assumptions(tr)

        if result:
            tr.intendedEvaluation = True
            if assum:
                pos_genuine.append(tr)
            else:
                pos_vacuous.append(tr)
        else:
            tr.intendedEvaluation = False
            if assum:
                neg_assum_hold.append(tr)
            else:
                neg_assum_fail.append(tr)

        if (len(neg_assum_hold) >= num_neg * 2 and
            len(pos_genuine) >= num_pos and
            len(pos_vacuous) >= num_pos):
            break

    # Prioritize assumption-satisfying negatives
    selected_neg = neg_assum_hold[:num_neg]
    if len(selected_neg) < num_neg:
        selected_neg += neg_assum_fail[:num_neg - len(selected_neg)]

    # Mix genuine and vacuous positives
    half = num_pos // 2
    selected_pos = pos_genuine[:half] + pos_vacuous[:half]
    remaining = num_pos - len(selected_pos)
    if remaining > 0:
        extras = pos_genuine[half:] + pos_vacuous[half:]
        selected_pos += extras[:remaining]

    return selected_pos, selected_neg


# ============================================================
# Runner with timeout (multiprocessing-based)
# ============================================================

def _run_gr1_worker(q, traces_data, max_depth, max_justices, max_guarantees):
    """Worker process for GR(1) miner with template enumeration."""
    from experiments.gr1_experiment import generate_template_configs, verify_formula
    traces = traces_data
    start = time.time()
    result_formula = None
    result_depth = None
    result_config = None

    configs = generate_template_configs(max_justices, max_guarantees)

    from z3 import sat
    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            encoder = GR1SATEncoding(D, traces,
                                     num_justices=m,
                                     num_guarantees=k,
                                     include_init=include_init,
                                     include_safety=include_safety)
            encoder.encodeFormula()
            if encoder.solver.check() == sat:
                model = encoder.solver.model()
                formula = encoder.reconstructWholeFormula(model)
                # Soundness check
                if verify_formula(formula, traces):
                    result_formula = formula
                    result_depth = D
                    result_config = (include_init, include_safety, m, k)
                    elapsed = time.time() - start
                    q.put(('gr1', result_formula, result_depth, elapsed, result_config))
                    return

    elapsed = time.time() - start
    q.put(('gr1', result_formula, result_depth, elapsed, result_config))


def _run_baseline_worker(q, traces_data, max_depth):
    """Worker process for baseline LTL miner."""
    traces = traces_data
    # Give baseline all operators
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    start = time.time()
    result_formula = None
    result_depth = None

    for D in range(1, max_depth + 1):
        encoder = DagSATEncoding(D, traces)
        encoder.encodeFormula()
        from z3 import sat
        if encoder.solver.check() == sat:
            model = encoder.solver.model()
            result_formula = encoder.reconstructWholeFormula(model)
            result_depth = D
            break

    elapsed = time.time() - start
    q.put(('baseline', result_formula, result_depth, elapsed))


def _run_dt_worker(q, traces_data):
    """Worker process for DT (decision tree) baseline."""
    traces = traces_data
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    try:
        result = run_dt_solver(traces, encoder=DagSATEncoding)
        # result is [timePassed, numAtoms, numPrimitives]
        q.put(('dt', result[0], result[1], result[2]))
    except Exception as e:
        q.put(('dt', None, None, None))


def run_with_timeout(worker_fn, args, timeout=TIMEOUT):
    """Run a worker function with a timeout. Returns whatever the worker puts in the queue."""
    q = Queue()
    p = Process(target=worker_fn, args=(q, *args))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # timed out

    if not q.empty():
        return q.get()
    else:
        return None


# ============================================================
# Main evaluation
# ============================================================

def main():
    benchmarks = make_benchmarks()
    results = []

    print("=" * 100)
    print("GR(1) Mining Evaluation — %d benchmarks, %ds timeout" % (len(benchmarks), TIMEOUT))
    print("=" * 100)
    print()

    header = "%-22s | %4s | %3s | %4s | %2s | %2s | %4s | %10s | %5s | %10s | %5s | %10s | %8s" % (
        'Benchmark', 'Vars', 'D_e', 'Exp', 'nJ', 'nG', 'Found',
        'GR1_time', 'GR1_D', 'LTL_time', 'LTL_D', 'DT_time', 'Speedup')
    print(header)
    print("-" * len(header))

    for bm in benchmarks:
        name = bm['name']
        spec = bm['spec']
        num_vars = bm['num_vars']
        expected_D = bm['expected_D']
        nj_expected = bm['num_justices']
        ng_expected = bm['num_guarantees']
        inc_init_expected = bm['include_init']
        inc_safety_expected = bm['include_safety']

        expected_template = ''
        if inc_init_expected: expected_template += 'I'
        if inc_safety_expected: expected_template += 'S'
        expected_template += 'J'

        # Generate traces
        pos_traces, neg_traces = generate_traces(spec, num_vars, num_pos=15, num_neg=10,
                                                  trace_length=5, seed=hash(name) % 10000)

        if len(neg_traces) < 3:
            print("%-22s | SKIP — insufficient negative traces (%d)" % (name, len(neg_traces)))
            continue

        traces_gr1 = ExperimentTraces(
            tracesToAccept=pos_traces, tracesToReject=neg_traces,
            operators=['&', '|', '!']
        )
        traces_ltl = ExperimentTraces(
            tracesToAccept=list(pos_traces), tracesToReject=list(neg_traces),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )

        # Run GR(1) miner — enumerates over template configurations
        gr1_max_depth = max(expected_D + 2, 5)
        gr1_result = run_with_timeout(
            _run_gr1_worker,
            (traces_gr1, gr1_max_depth, 3, 3),
            timeout=TIMEOUT
        )

        if gr1_result is not None:
            _, gr1_formula, gr1_depth, gr1_time, gr1_config = gr1_result
            gr1_timeout = False
            if gr1_config is not None:
                inc_init, inc_safety, nj, ng = gr1_config
                found_template = ''
                if inc_init: found_template += 'I'
                if inc_safety: found_template += 'S'
                found_template += 'J'
            else:
                nj, ng, found_template = '-', '-', '-'
        else:
            gr1_formula, gr1_depth, gr1_time = None, None, TIMEOUT
            gr1_timeout = True
            nj, ng, found_template = '-', '-', '-'

        # Run baseline LTL miner
        ltl_max_depth = 10
        ltl_result = run_with_timeout(
            _run_baseline_worker,
            (traces_ltl, ltl_max_depth),
            timeout=TIMEOUT
        )

        if ltl_result is not None:
            _, ltl_formula, ltl_depth, ltl_time = ltl_result
            ltl_timeout = False
        else:
            ltl_formula, ltl_depth, ltl_time = None, None, TIMEOUT
            ltl_timeout = True

        # Run DT baseline
        traces_dt = ExperimentTraces(
            tracesToAccept=list(pos_traces), tracesToReject=list(neg_traces),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )
        dt_result = run_with_timeout(
            _run_dt_worker,
            (traces_dt,),
            timeout=TIMEOUT
        )

        if dt_result is not None:
            _, dt_time, dt_num_atoms, dt_num_primitives = dt_result
            dt_timeout = dt_time is None
            if dt_timeout:
                dt_time = TIMEOUT
        else:
            dt_time, dt_num_atoms, dt_num_primitives = TIMEOUT, None, None
            dt_timeout = True

        # Format results
        gr1_time_str = 'TIMEOUT' if gr1_timeout else '%.3fs' % gr1_time
        gr1_d_str = str(gr1_depth) if gr1_depth else '-'
        ltl_time_str = 'TIMEOUT' if ltl_timeout else '%.3fs' % ltl_time
        ltl_d_str = str(ltl_depth) if ltl_depth else '-'
        dt_time_str = 'TIMEOUT' if dt_timeout else '%.3fs' % dt_time

        if not gr1_timeout and not ltl_timeout and gr1_time > 0:
            speedup = '%.1fx' % (ltl_time / gr1_time)
        elif gr1_timeout:
            speedup = '-'
        elif ltl_timeout:
            speedup = '>%.0fx' % (TIMEOUT / gr1_time) if gr1_time > 0 else 'inf'
        else:
            speedup = '-'

        print("%-22s | %4d | %3d | %4s | %2s | %2s | %4s | %10s | %5s | %10s | %5s | %10s | %8s" % (
            name, num_vars, expected_D, expected_template, str(nj), str(ng), found_template,
            gr1_time_str, gr1_d_str, ltl_time_str, ltl_d_str, dt_time_str, speedup))

        results.append({
            'benchmark': name,
            'num_vars': num_vars,
            'expected_D': expected_D,
            'expected_template': expected_template,
            'found_template': found_template,
            'found_justices': nj,
            'found_guarantees': ng,
            'gr1_time': gr1_time if not gr1_timeout else None,
            'gr1_depth': gr1_depth,
            'gr1_timeout': gr1_timeout,
            'ltl_time': ltl_time if not ltl_timeout else None,
            'ltl_depth': ltl_depth,
            'ltl_timeout': ltl_timeout,
            'dt_time': dt_time if not dt_timeout else None,
            'dt_num_atoms': dt_num_atoms,
            'dt_num_primitives': dt_num_primitives,
            'dt_timeout': dt_timeout,
            'gr1_formula': str(gr1_formula) if gr1_formula else None,
            'ltl_formula': (ltl_formula.prettyPrint(True) if ltl_formula else None),
        })

    # Write CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Results saved to %s" % os.path.abspath(csv_path))

    # Summary
    gr1_solved = sum(1 for r in results if not r['gr1_timeout'] and r['gr1_depth'] is not None)
    ltl_solved = sum(1 for r in results if not r['ltl_timeout'] and r['ltl_depth'] is not None)
    dt_solved = sum(1 for r in results if not r['dt_timeout'])
    print("\nSummary: GR(1) solved %d/%d, SAT Baseline solved %d/%d, DT Baseline solved %d/%d" % (
        gr1_solved, len(results), ltl_solved, len(results), dt_solved, len(results)))


if __name__ == '__main__':
    main()
