"""Compare GR1Mine vs samples2LTL[SAT] with restricted operators (no U, no ->)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.Traces import ExperimentTraces
from utils.spectra_parser import parse_spectra_file
from utils.trace_generator import generate_traces_sat
from experiments.eval_spectra import generate_traces_from_spec
from experiments.gr1_experiment import run_gr1_incremental_solver
from smtEncoding.dagSATEncoding import DagSATEncoding
from z3 import sat
import time


SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

BENCHMARKS = [
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_2.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_trans_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_wo_ass_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_wo_ass_fairness_amba_ahb_2.spectra',
]


def mine_sat(traces, operators, max_depth=10, timeout=60):
    """Mine LTL formula with given operator set."""
    traces.operators = operators
    start = time.time()
    for D in range(1, max_depth + 1):
        if time.time() - start > timeout:
            return None, None, time.time() - start
        print("    SAT D=%d ops=%s ..." % (D, operators[:3]), end=' ', flush=True)
        encoder = DagSATEncoding(D, traces)
        encoder.encodeFormula()
        if encoder.solver.check() == sat:
            model = encoder.solver.model()
            formula = encoder.reconstructWholeFormula(model)
            elapsed = time.time() - start
            print("SAT (%.1fs)" % elapsed, flush=True)
            return formula, D, elapsed
        print("UNSAT (%.1fs)" % (time.time() - start), flush=True)
    return None, None, time.time() - start


if __name__ == '__main__':
    print("=" * 150)
    print("GR1Mine vs SAT[full ops] vs SAT[restricted: G,F,!,&,|,X]")
    print("=" * 150)
    print()
    print("%-40s | %4s | %10s %5s %-30s | %10s %5s %-30s | %10s %5s %-30s" % (
        'Benchmark', 'Vars',
        'GR1_time', 'D', 'GR1 formula',
        'SAT_time', 'D', 'SAT formula (full)',
        'SAT_r_time', 'D', 'SAT formula (restricted)'))
    print("-" * 150)

    for bm in BENCHMARKS:
        path = os.path.join(SPECTRA_DIR, bm)
        if not os.path.exists(path):
            continue

        gr1_gt, meta = parse_spectra_file(path)
        n = meta['num_vars']
        nj = len(gr1_gt.justices)
        ng = len(gr1_gt.guarantees)
        name = os.path.basename(bm).replace('.spectra', '')

        # Generate training traces: mix of SAT-based + random
        sat_pos, sat_neg = generate_traces_sat(
            gr1_gt, n, num_pos=10, num_neg=15,
            trace_lengths=[4, 5, 6], max_trials_per_config=20
        )
        rand_pos, rand_neg = generate_traces_from_spec(
            gr1_gt, n, num_pos=15, num_neg=10,
            trace_length=5, max_trials=50000,
            seed=hash(name) % 10000
        )
        pos = sat_pos + rand_pos
        neg = sat_neg + rand_neg
        if len(neg) < 3:
            print("%-40s | %4d | SKIP — %d neg traces" % (name, n, len(neg)))
            continue

        num_train = len(pos) + len(neg)

        # --- GR1Mine (incremental) ---
        traces_gr1 = ExperimentTraces(
            tracesToAccept=list(pos), tracesToReject=list(neg),
            operators=['&', '|', '!']
        )
        gr1_result = run_gr1_incremental_solver(
            traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4))
        gr1_f, gr1_d, gr1_t = gr1_result[0], gr1_result[1], gr1_result[2]

        # --- SAT baseline (full operators) ---
        traces_full = ExperimentTraces(
            tracesToAccept=list(pos), tracesToReject=list(neg),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )
        sat_f, sat_d, sat_t = mine_sat(traces_full, ['G', 'F', '!', 'U', '&', '|', '->', 'X'], max_depth=7)

        # --- SAT baseline (restricted: no U, no ->) ---
        traces_restr = ExperimentTraces(
            tracesToAccept=list(pos), tracesToReject=list(neg),
            operators=['G', 'F', '!', '&', '|', 'X']
        )
        satr_f, satr_d, satr_t = mine_sat(traces_restr, ['G', 'F', '!', '&', '|', 'X'], max_depth=7)

        # Format
        def fmt(f, d, t):
            if f is None:
                return '%10s %5s %-30s' % ('TIMEOUT', '-', '')
            fs = f.prettyPrint(True) if hasattr(f, 'prettyPrint') else str(f)
            if len(fs) > 28:
                fs = fs[:25] + '...'
            return '%9.2fs %5s %-30s' % (t, d, fs)

        gr1_fs = gr1_f.prettyPrint() if gr1_f else 'FAIL'
        if len(gr1_fs) > 28:
            gr1_fs = gr1_fs[:25] + '...'
        gr1_fmt = '%9.2fs %5s %-30s' % (gr1_t, gr1_d, gr1_fs)

        print("%-40s | %4d | %s | %s | %s" % (
            name, n, gr1_fmt, fmt(sat_f, sat_d, sat_t), fmt(satr_f, satr_d, satr_t)))
