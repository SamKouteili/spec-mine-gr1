"""Quick benchmark: compare original vs incremental GR1 miner on Spectra benchmarks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.Traces import ExperimentTraces
from utils.spectra_parser import parse_spectra_file
from experiments.eval_spectra import generate_traces_from_spec
from experiments.gr1_experiment import run_gr1_enumerating_solver, run_gr1_incremental_solver

SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

BENCHMARKS = [
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_2.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_trans_amba_ahb_1.spectra',
    'cimattiAnalyzing/gen_buf_w_guar_fairness_5_genbuf.spectra',
]


def make_traces(spec_path):
    gr1, meta = parse_spectra_file(spec_path)
    n = meta['num_vars']
    nj = len(gr1.justices)
    ng = len(gr1.guarantees)

    pos, neg = generate_traces_from_spec(
        gr1, n, num_pos=15, num_neg=10,
        trace_length=5, max_trials=50000,
        seed=hash(os.path.basename(spec_path)) % 10000
    )
    if len(neg) < 3:
        return None, nj, ng, n

    traces = ExperimentTraces(
        tracesToAccept=pos, tracesToReject=neg,
        operators=['&', '|', '!']
    )
    return traces, nj, ng, n


if __name__ == '__main__':
    print("%-55s | %4s | %2s | %2s | %10s | %10s | %6s" % (
        'Benchmark', 'Vars', 'nJ', 'nG', 'Original', 'Incremental', 'Speedup'))
    print("-" * 110)

    for bm in BENCHMARKS:
        path = os.path.join(SPECTRA_DIR, bm)
        if not os.path.exists(path):
            print("%-55s | SKIP — file not found" % bm)
            continue

        traces, nj, ng, n = make_traces(path)
        if traces is None:
            print("%-55s | %4d | %2d | %2d | SKIP — too few neg traces" % (bm, n, nj, ng))
            continue

        max_j = min(nj + 1, 4)
        max_g = min(ng + 1, 4)

        # Original solver
        r1 = run_gr1_enumerating_solver(traces, 3, max_j, max_g)
        t_orig = r1[2]

        # Incremental solver
        r2 = run_gr1_incremental_solver(traces, 3, max_j, max_g)
        t_incr = r2[2]

        speedup = t_orig / t_incr if t_incr > 0 else float('inf')

        print("%-55s | %4d | %2d | %2d | %9.3fs | %9.3fs | %5.2fx" % (
            bm, n, nj, ng, t_orig, t_incr, speedup))
