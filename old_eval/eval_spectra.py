"""
Evaluation on Spectra/Syntech GR(1) benchmarks.

Parses real GR(1) specs from the Spectra benchmark suite, generates
traces, and runs GR1Mine + baseline SAT + baseline DT with 5-min timeout.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import csv
import time
import glob
from multiprocessing import Process, Queue

from utils.SimpleTree import Formula
from utils.Traces import Trace, ExperimentTraces
from utils.GR1Formula import GR1Formula
from utils.spectra_parser import parse_spectra_file
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from smtEncoding.dagSATEncoding import DagSATEncoding
from solverRuns import run_dt_solver


TIMEOUT = 300  # 5 minutes


# ============================================================
# Trace generation
# ============================================================

def generate_traces_from_spec(spec, num_vars, num_pos=15, num_neg=10,
                              trace_length=5, max_trials=20000, seed=42):
    """
    Generate structured traces that prevent spurious separators.

    Strategy: partition traces by whether assumptions hold, and prioritize
    assumption-satisfying negatives (the most constraining kind — they force
    any correct separator to get the guarantee side right).
    """
    random.seed(seed)

    # Buckets: neg where assumptions hold, neg where they don't,
    #          pos where assumptions hold (genuine), pos where they fail (vacuous)
    neg_assum_hold = []
    neg_assum_fail = []
    pos_genuine = []
    pos_vacuous = []

    for _ in range(max_trials):
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

        # Stop early if we have plenty of each kind
        if (len(neg_assum_hold) >= num_neg * 2 and
            len(pos_genuine) >= num_pos and
            len(pos_vacuous) >= num_pos):
            break

    # Assemble: prioritize assumption-satisfying negatives
    selected_neg = neg_assum_hold[:num_neg]
    # Fill remaining with assumption-failing negatives if needed
    if len(selected_neg) < num_neg:
        selected_neg += neg_assum_fail[:num_neg - len(selected_neg)]

    # Mix of genuine and vacuous positives
    half = num_pos // 2
    selected_pos = pos_genuine[:half] + pos_vacuous[:half]
    # Fill up from whichever has more
    remaining = num_pos - len(selected_pos)
    if remaining > 0:
        extras = pos_genuine[half:] + pos_vacuous[half:]
        selected_pos += extras[:remaining]

    return selected_pos, selected_neg


# ============================================================
# Workers
# ============================================================

def _run_gr1_worker(q, traces_data, max_depth, max_justices, max_guarantees,
                    env_var_indices=None):
    """Worker for GR(1) miner with template enumeration."""
    from experiments.gr1_experiment import generate_template_configs, verify_formula
    from z3 import sat
    traces = traces_data
    start = time.time()

    configs = generate_template_configs(max_justices, max_guarantees)

    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            encoder = GR1SATEncoding(D, traces,
                                     num_justices=m,
                                     num_guarantees=k,
                                     include_init=include_init,
                                     include_safety=include_safety,
                                     env_var_indices=env_var_indices)
            encoder.encodeFormula()
            if encoder.solver.check() == sat:
                model = encoder.solver.model()
                formula = encoder.reconstructWholeFormula(model)
                if verify_formula(formula, traces):
                    elapsed = time.time() - start
                    config = (include_init, include_safety, m, k)
                    q.put(('gr1', formula, D, elapsed, config))
                    return

    elapsed = time.time() - start
    q.put(('gr1', None, None, elapsed, None))


def _run_sat_worker(q, traces_data, max_depth):
    """Worker for baseline SAT LTL miner."""
    from z3 import sat
    traces = traces_data
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    start = time.time()

    for D in range(1, max_depth + 1):
        encoder = DagSATEncoding(D, traces)
        encoder.encodeFormula()
        if encoder.solver.check() == sat:
            model = encoder.solver.model()
            formula = encoder.reconstructWholeFormula(model)
            elapsed = time.time() - start
            q.put(('sat', formula, D, elapsed))
            return

    elapsed = time.time() - start
    q.put(('sat', None, None, elapsed))


def _run_dt_worker(q, traces_data):
    """Worker for DT baseline."""
    traces = traces_data
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    try:
        result = run_dt_solver(traces, encoder=DagSATEncoding)
        # result is [timePassed, numAtoms, numPrimitives]
        q.put(('dt', result[0], result[1], result[2]))
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        q.put(('dt', None, None, None))


def run_with_timeout(worker_fn, args, timeout=TIMEOUT):
    """Run a worker with timeout."""
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
# Benchmark selection
# ============================================================

def find_spectra_benchmarks(spectra_dir, max_vars=None):
    """Find and parse Spectra benchmarks, optionally filtering by var count."""
    benchmarks = []
    patterns = [
        os.path.join(spectra_dir, 'bloemDebugging', '*.spectra'),
        os.path.join(spectra_dir, 'cimattiAnalyzing', '*.spectra'),
    ]

    for pattern in patterns:
        for filepath in sorted(glob.glob(pattern)):
            try:
                gr1, meta = parse_spectra_file(filepath)
                n = meta['num_vars']
                if max_vars is not None and n > max_vars:
                    continue
                benchmarks.append({
                    'name': os.path.basename(filepath).replace('.spectra', ''),
                    'path': filepath,
                    'spec': gr1,
                    'num_vars': n,
                    'num_justices': len(gr1.justices),
                    'num_guarantees': len(gr1.guarantees),
                    'has_init': gr1.init_e is not None or gr1.init_s is not None,
                    'has_safety': gr1.safety_e is not None or gr1.safety_s is not None,
                    'metadata': meta,
                })
            except Exception as e:
                print("SKIP %s: %s" % (os.path.basename(filepath), e))

    return benchmarks


# ============================================================
# Main
# ============================================================

def main():
    spectra_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')
    if not os.path.isdir(spectra_dir):
        print("ERROR: spectra-specs directory not found at %s" % spectra_dir)
        return

    # Find benchmarks — start with all boolean-only specs
    benchmarks = find_spectra_benchmarks(spectra_dir, max_vars=30)
    print("=" * 120)
    print("Spectra Benchmark Evaluation — %d specs, %ds timeout" % (len(benchmarks), TIMEOUT))
    print("=" * 120)
    print()

    header = "%-45s | %4s | %2s | %2s | %4s | %10s | %5s | %10s | %5s | %10s" % (
        'Benchmark', 'Vars', 'nJ', 'nG', 'Tmpl',
        'GR1_time', 'GR1_D', 'SAT_time', 'SAT_D', 'DT_time')
    print(header)
    print("-" * len(header))

    results = []

    for bm in benchmarks:
        name = bm['name']
        spec = bm['spec']
        num_vars = bm['num_vars']
        nj = bm['num_justices']
        ng = bm['num_guarantees']

        tmpl = ''
        if bm['has_init']:
            tmpl += 'I'
        if bm['has_safety']:
            tmpl += 'S'
        tmpl += 'J'

        # Generate traces
        pos_traces, neg_traces = generate_traces_from_spec(
            spec, num_vars, num_pos=15, num_neg=10,
            trace_length=5, max_trials=50000,
            seed=hash(name) % 10000
        )

        if len(neg_traces) < 3:
            print("%-45s | %4d | SKIP — %d neg traces" % (name, num_vars, len(neg_traces)))
            continue

        traces_gr1 = ExperimentTraces(
            tracesToAccept=pos_traces, tracesToReject=neg_traces,
            operators=['&', '|', '!']
        )

        # Compute env variable indices from metadata
        meta = bm['metadata']
        env_var_indices = [meta['name_to_index'][v] for v in meta['env_vars']]

        # --- GR(1) miner ---
        gr1_result = run_with_timeout(
            _run_gr1_worker,
            (traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4), env_var_indices),
            timeout=TIMEOUT
        )

        if gr1_result is not None:
            _, gr1_formula, gr1_depth, gr1_time, gr1_config = gr1_result
            gr1_timeout = gr1_formula is None
            if gr1_timeout:
                gr1_time = TIMEOUT
        else:
            gr1_formula, gr1_depth, gr1_time = None, None, TIMEOUT
            gr1_timeout = True

        # --- SAT baseline ---
        traces_sat = ExperimentTraces(
            tracesToAccept=list(pos_traces), tracesToReject=list(neg_traces),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )
        sat_result = run_with_timeout(
            _run_sat_worker,
            (traces_sat, 10),
            timeout=TIMEOUT
        )

        if sat_result is not None:
            _, sat_formula, sat_depth, sat_time = sat_result
            sat_timeout = sat_formula is None
            if sat_timeout:
                sat_time = TIMEOUT
        else:
            sat_formula, sat_depth, sat_time = None, None, TIMEOUT
            sat_timeout = True

        # --- DT baseline ---
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
            _, dt_time, dt_atoms, dt_prims = dt_result
            dt_timeout = dt_time is None
            if dt_timeout:
                dt_time = TIMEOUT
        else:
            dt_time = TIMEOUT
            dt_timeout = True

        # Format
        gr1_str = 'TIMEOUT' if gr1_timeout else '%.3fs' % gr1_time
        gr1_d = str(gr1_depth) if gr1_depth else '-'
        sat_str = 'TIMEOUT' if sat_timeout else '%.3fs' % sat_time
        sat_d = str(sat_depth) if sat_depth else '-'
        dt_str = 'TIMEOUT' if dt_timeout else '%.3fs' % dt_time

        print("%-45s | %4d | %2d | %2d | %4s | %10s | %5s | %10s | %5s | %10s" % (
            name, num_vars, nj, ng, tmpl,
            gr1_str, gr1_d, sat_str, sat_d, dt_str))

        results.append({
            'benchmark': name,
            'num_vars': num_vars,
            'num_justices': nj,
            'num_guarantees': ng,
            'template': tmpl,
            'gr1_time': gr1_time if not gr1_timeout else None,
            'gr1_depth': gr1_depth,
            'gr1_timeout': gr1_timeout,
            'sat_time': sat_time if not sat_timeout else None,
            'sat_depth': sat_depth,
            'sat_timeout': sat_timeout,
            'dt_time': dt_time if not dt_timeout else None,
            'dt_timeout': dt_timeout,
            'gr1_formula': str(gr1_formula) if gr1_formula else None,
            'sat_formula': (sat_formula.prettyPrint(True) if sat_formula else None),
        })

    # Write CSV
    if results:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'spectra_eval_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print()
        print("Results saved to %s" % os.path.abspath(csv_path))

    # Summary
    total = len(results)
    gr1_solved = sum(1 for r in results if not r['gr1_timeout'])
    sat_solved = sum(1 for r in results if not r['sat_timeout'])
    dt_solved = sum(1 for r in results if not r['dt_timeout'])
    print("\nSummary (%d benchmarks): GR1 %d/%d, SAT %d/%d, DT %d/%d" % (
        total, gr1_solved, total, sat_solved, total, dt_solved, total))


if __name__ == '__main__':
    main()
