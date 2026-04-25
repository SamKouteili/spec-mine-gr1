"""
Full evaluation: Mining + Synthesis + Generalization on Spectra benchmarks.

For each benchmark:
1. Parse ground-truth GR(1) spec, generate training + held-out traces
2. Mine with GR1Mine (partitioned), samples2LTL[SAT], samples2LTL[DT]
3. Check realizability: GR1Mine via slugs, SAT/DT via ltlsynt
4. Evaluate mined formula on held-out traces (generalization accuracy)

Traces: 25 pos + 25 neg, mixed SAT-generated + random for diverse coverage.
"""
import sys, os, subprocess, time, random, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from z3 import sat as Z3_SAT
from utils.spectra_parser import parse_spectra_file
from utils.Traces import Trace, ExperimentTraces
from utils.trace_generator import generate_traces_sat
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from smtEncoding.dagSATEncoding import DagSATEncoding
from experiments.gr1_experiment import generate_template_configs, verify_formula
from experiments.synthesis_comparison import gr1_to_slugs, run_slugs_synthesis
from solverRuns import run_dt_solver


SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

MINE_TIMEOUT = 300   # 5 min for mining
SYNTH_TIMEOUT = 60   # 1 min for synthesis

TRAIN_POS = 25
TRAIN_NEG = 25
HELDOUT_POS = 100
HELDOUT_NEG = 100


# ============================================================
# Trace generation (mixed SAT + random)
# ============================================================

def generate_random_traces(spec, num_vars, num_pos, num_neg,
                           trace_length=5, max_trials=50000, seed=42):
    """Generate random traces, classified by ground-truth spec."""
    random.seed(seed)
    neg_assum_hold = []
    neg_assum_fail = []
    pos_genuine = []
    pos_vacuous = []

    for _ in range(max_trials):
        length = random.randint(max(2, trace_length - 1), trace_length + 1)
        lasso = random.randint(0, length - 1)
        vec = [[bool(random.randint(0, 1)) for _ in range(num_vars)]
               for _ in range(length)]
        tr = Trace(vec, lasso)

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

    selected_neg = neg_assum_hold[:num_neg]
    if len(selected_neg) < num_neg:
        selected_neg += neg_assum_fail[:num_neg - len(selected_neg)]

    half = num_pos // 2
    selected_pos = pos_genuine[:half] + pos_vacuous[:half]
    remaining = num_pos - len(selected_pos)
    if remaining > 0:
        extras = pos_genuine[half:] + pos_vacuous[half:]
        selected_pos += extras[:remaining]

    return selected_pos, selected_neg


def generate_mixed_traces(spec, num_vars, num_pos, num_neg, seed=42):
    """Generate a mix of SAT-constructed + random traces.

    Half from SAT-based generation (guarantees assumption satisfaction),
    half from random sampling (for diversity).
    """
    half_pos = num_pos // 2
    half_neg = num_neg // 2

    # SAT-generated traces
    sat_pos, sat_neg = generate_traces_sat(
        spec, num_vars,
        num_pos=half_pos, num_neg=half_neg,
        trace_lengths=[4, 5, 6],
        max_trials_per_config=20
    )

    # Random traces
    rand_pos, rand_neg = generate_random_traces(
        spec, num_vars,
        num_pos=num_pos - len(sat_pos),
        num_neg=num_neg - len(sat_neg),
        seed=seed
    )

    pos = sat_pos + rand_pos
    neg = sat_neg + rand_neg
    return pos[:num_pos], neg[:num_neg]


# ============================================================
# Mining (in-process with wall-clock timeout)
# ============================================================

def mine_gr1(traces, max_depth, max_justices, max_guarantees,
             env_var_indices, timeout=MINE_TIMEOUT):
    """Mine with GR1Mine (partitioned)."""
    start = time.time()
    configs = generate_template_configs(max_justices, max_guarantees)

    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            if time.time() - start > timeout:
                return None, None, timeout, None

            encoder = GR1SATEncoding(D, traces,
                                     num_justices=m, num_guarantees=k,
                                     include_init=include_init,
                                     include_safety=include_safety,
                                     env_var_indices=env_var_indices)
            encoder.encodeFormula()

            if encoder.solver.check() == Z3_SAT:
                model = encoder.solver.model()
                formula = encoder.reconstructWholeFormula(model)
                if verify_formula(formula, traces):
                    return formula, D, time.time() - start, (include_init, include_safety, m, k)

    return None, None, time.time() - start, None


def mine_sat(traces, max_depth, timeout=MINE_TIMEOUT):
    """Mine with samples2LTL[SAT]."""
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    start = time.time()

    for D in range(1, max_depth + 1):
        if time.time() - start > timeout:
            return None, None, timeout

        encoder = DagSATEncoding(D, traces)
        encoder.encodeFormula()

        if encoder.solver.check() == Z3_SAT:
            model = encoder.solver.model()
            formula = encoder.reconstructWholeFormula(model)
            return formula, D, time.time() - start

    return None, None, time.time() - start


def mine_dt(traces):
    """Mine with samples2LTL[DT]."""
    traces.operators = ['G', 'F', '!', 'U', '&', '|', '->', 'X']
    try:
        result = run_dt_solver(traces, encoder=DagSATEncoding)
        # result = [timePassed, numAtoms, numPrimitives]
        return result[0], result[1], result[2]
    except Exception:
        return None, None, None


# ============================================================
# Synthesis
# ============================================================

def synthesize_gr1_slugs(gr1_formula, env_vars, sys_vars, all_vars):
    """Synthesize GR1Mine output via slugs."""
    slugs_input = gr1_to_slugs(gr1_formula, env_vars, sys_vars, all_vars)
    realizable, synth_time, msg = run_slugs_synthesis(slugs_input, timeout=SYNTH_TIMEOUT)
    return realizable, synth_time, msg


def formula_to_ltl_str(formula, all_vars):
    """Convert a SimpleTree Formula to an ltlsynt-compatible LTL string."""
    label = formula.label
    if label.startswith('x'):
        primed = label.endswith("'")
        idx_str = label[1:-1] if primed else label[1:]
        idx = int(idx_str)
        var_name = all_vars[idx] if idx < len(all_vars) else label
        if primed:
            return 'X(%s)' % var_name
        return var_name
    elif label == '&':
        return '(%s & %s)' % (formula_to_ltl_str(formula.left, all_vars),
                               formula_to_ltl_str(formula.right, all_vars))
    elif label == '|':
        return '(%s | %s)' % (formula_to_ltl_str(formula.left, all_vars),
                               formula_to_ltl_str(formula.right, all_vars))
    elif label == '!':
        return '!(%s)' % formula_to_ltl_str(formula.left, all_vars)
    elif label == '->':
        return '(%s -> %s)' % (formula_to_ltl_str(formula.left, all_vars),
                                formula_to_ltl_str(formula.right, all_vars))
    elif label == 'G':
        return 'G(%s)' % formula_to_ltl_str(formula.left, all_vars)
    elif label == 'F':
        return 'F(%s)' % formula_to_ltl_str(formula.left, all_vars)
    elif label == 'X':
        return 'X(%s)' % formula_to_ltl_str(formula.left, all_vars)
    elif label == 'U':
        return '(%s U %s)' % (formula_to_ltl_str(formula.left, all_vars),
                               formula_to_ltl_str(formula.right, all_vars))
    else:
        return label


def synthesize_ltl_ltlsynt(formula, env_vars, sys_vars, all_vars):
    """Synthesize flat LTL formula via ltlsynt.

    Note: spot/ltlsynt treats uppercase identifiers as LTL operators,
    so all variable names must be lowercased.
    """
    # Lowercase all variable names for ltlsynt compatibility
    all_vars_lc = [v.lower() for v in all_vars]
    env_vars_lc = [v.lower() for v in env_vars]
    sys_vars_lc = [v.lower() for v in sys_vars]
    ltl_str = formula_to_ltl_str(formula, all_vars_lc)
    try:
        start = time.time()
        result = subprocess.run(
            ['ltlsynt',
             '--ins=' + ','.join(env_vars_lc),
             '--outs=' + ','.join(sys_vars_lc),
             '-f', ltl_str],
            capture_output=True, text=True, timeout=SYNTH_TIMEOUT
        )
        elapsed = time.time() - start
        if 'REALIZABLE' in result.stdout and 'UNREALIZABLE' not in result.stdout:
            return True, elapsed, 'REALIZABLE'
        elif 'UNREALIZABLE' in result.stdout:
            return False, elapsed, 'UNREALIZABLE'
        else:
            return None, elapsed, result.stderr.strip()[:100]
    except subprocess.TimeoutExpired:
        return None, SYNTH_TIMEOUT, 'TIMEOUT'


# ============================================================
# Generalization (formula-level evaluation on held-out traces)
# ============================================================

def eval_generalization_gr1(gr1_formula, heldout_pos, heldout_neg):
    """Evaluate GR1Formula on held-out traces."""
    tp = tn = fp = fn = 0
    for tr in heldout_pos:
        if gr1_formula.evaluate_on_trace(tr):
            tp += 1
        else:
            fn += 1
    for tr in heldout_neg:
        if gr1_formula.evaluate_on_trace(tr):
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    return acc, tp, tn, fp, fn


def eval_generalization_ltl(formula, heldout_pos, heldout_neg):
    """Evaluate a SimpleTree LTL Formula on held-out traces."""
    tp = tn = fp = fn = 0
    for tr in heldout_pos:
        if tr.evaluateFormulaOnTrace(formula):
            tp += 1
        else:
            fn += 1
    for tr in heldout_neg:
        if tr.evaluateFormulaOnTrace(formula):
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    return acc, tp, tn, fp, fn


# ============================================================
# Benchmark discovery
# ============================================================

def find_benchmarks(max_vars=30):
    """Find parseable Spectra benchmarks."""
    import glob as globmod
    benchmarks = []
    patterns = [
        os.path.join(SPECTRA_DIR, 'bloemDebugging', '*.spectra'),
        os.path.join(SPECTRA_DIR, 'cimattiAnalyzing', '*.spectra'),
    ]
    for pattern in sorted(patterns):
        for filepath in sorted(globmod.glob(pattern)):
            try:
                gr1, meta = parse_spectra_file(filepath)
                n = meta['num_vars']
                if n > max_vars:
                    continue
                benchmarks.append({
                    'name': os.path.basename(filepath).replace('.spectra', ''),
                    'path': filepath,
                    'spec': gr1,
                    'metadata': meta,
                })
            except Exception:
                pass
    return benchmarks


# ============================================================
# Main
# ============================================================

def main():
    benchmarks = find_benchmarks(max_vars=30)
    print("=" * 140)
    print("Full Evaluation: Mining + Synthesis + Generalization  (%d benchmarks)" % len(benchmarks))
    print("Traces: %d pos + %d neg (mixed SAT + random)  |  Held-out: %d pos + %d neg" % (
        TRAIN_POS, TRAIN_NEG, HELDOUT_POS, HELDOUT_NEG))
    print("Synthesis: GR1Mine -> slugs  |  SAT/DT -> ltlsynt")
    print("=" * 140)

    results = []

    for bm in benchmarks:
        name = bm['name']
        spec = bm['spec']
        meta = bm['metadata']
        num_vars = meta['num_vars']
        env_vars = meta['env_vars']
        sys_vars = meta['sys_vars']
        all_vars = meta['all_vars']
        env_indices = [meta['name_to_index'][v] for v in env_vars]
        nj = len(spec.justices)
        ng = len(spec.guarantees)

        print("\n" + "-" * 140)
        print("%-50s  vars=%d (env=%d, sys=%d)  J=%d G=%d" % (
            name, num_vars, len(env_vars), len(sys_vars), nj, ng))

        # --- Generate mixed training traces ---
        train_pos, train_neg = generate_mixed_traces(
            spec, num_vars, TRAIN_POS, TRAIN_NEG,
            seed=hash(name) % 10000)

        if len(train_neg) < 3:
            print("  SKIP — only %d negative traces" % len(train_neg))
            continue

        # --- Generate held-out traces (different seed, random only for speed) ---
        heldout_pos, heldout_neg = generate_random_traces(
            spec, num_vars, HELDOUT_POS, HELDOUT_NEG,
            seed=(hash(name) + 99999) % 1000000)

        print("  Training: %d pos, %d neg  |  Held-out: %d pos, %d neg" % (
            len(train_pos), len(train_neg), len(heldout_pos), len(heldout_neg)))

        row = {
            'benchmark': name,
            'num_vars': num_vars,
            'num_env': len(env_vars),
            'num_sys': len(sys_vars),
            'nJ': nj, 'nG': ng,
            'train_pos': len(train_pos), 'train_neg': len(train_neg),
            'heldout_pos': len(heldout_pos), 'heldout_neg': len(heldout_neg),
        }

        # ==================== GR1Mine (partitioned) ====================
        traces_gr1 = ExperimentTraces(
            tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
            operators=['&', '|', '!']
        )

        gr1_f, gr1_d, gr1_time, gr1_cfg = mine_gr1(
            traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4), env_indices)

        row['gr1_time'] = gr1_time
        row['gr1_depth'] = gr1_d
        row['gr1_formula'] = str(gr1_f) if gr1_f else None
        row['gr1_timeout'] = gr1_f is None

        if gr1_f:
            print("  GR1Mine:  D=%s  %.1fs  %s" % (gr1_d, gr1_time, str(gr1_f)[:90]))

            # Synthesis via slugs
            realizable, synth_t, msg = synthesize_gr1_slugs(gr1_f, env_vars, sys_vars, all_vars)
            row['gr1_realizable'] = realizable
            row['gr1_synth_time'] = synth_t
            status = 'REALIZABLE' if realizable else ('UNREALIZABLE' if realizable is False else 'TIMEOUT')
            print("    Synthesis (slugs): %s (%.3fs)" % (status, synth_t))

            # Generalization
            acc, tp, tn, fp, fn = eval_generalization_gr1(gr1_f, heldout_pos, heldout_neg)
            row['gr1_accuracy'] = acc
            row['gr1_tp'] = tp; row['gr1_tn'] = tn; row['gr1_fp'] = fp; row['gr1_fn'] = fn
            print("    Generalization: %.1f%% (%d/%d)  tp=%d tn=%d fp=%d fn=%d" % (
                acc * 100, tp + tn, tp + tn + fp + fn, tp, tn, fp, fn))
        else:
            print("  GR1Mine:  TIMEOUT (%.1fs)" % gr1_time)
            row.update({k: None for k in [
                'gr1_realizable', 'gr1_synth_time', 'gr1_accuracy',
                'gr1_tp', 'gr1_tn', 'gr1_fp', 'gr1_fn']})

        # ==================== samples2LTL[SAT] ====================
        traces_sat = ExperimentTraces(
            tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )

        sat_f, sat_d, sat_time = mine_sat(traces_sat, 10)

        row['sat_time'] = sat_time
        row['sat_depth'] = sat_d
        row['sat_timeout'] = sat_f is None

        if sat_f:
            sat_str = sat_f.prettyPrint(True)
            row['sat_formula'] = sat_str
            print("  SAT:      D=%s  %.1fs  %s" % (sat_d, sat_time, sat_str[:90]))

            # Synthesis via ltlsynt
            realizable, synth_t, msg = synthesize_ltl_ltlsynt(sat_f, env_vars, sys_vars, all_vars)
            row['sat_realizable'] = realizable
            row['sat_synth_time'] = synth_t
            status = 'REALIZABLE' if realizable else ('UNREALIZABLE' if realizable is False else msg[:20])
            print("    Synthesis (ltlsynt): %s (%.3fs)" % (status, synth_t))

            # Generalization
            acc, tp, tn, fp, fn = eval_generalization_ltl(sat_f, heldout_pos, heldout_neg)
            row['sat_accuracy'] = acc
            row['sat_tp'] = tp; row['sat_tn'] = tn; row['sat_fp'] = fp; row['sat_fn'] = fn
            print("    Generalization: %.1f%% (%d/%d)  tp=%d tn=%d fp=%d fn=%d" % (
                acc * 100, tp + tn, tp + tn + fp + fn, tp, tn, fp, fn))
        else:
            print("  SAT:      TIMEOUT (%.1fs)" % sat_time)
            row.update({k: None for k in [
                'sat_formula', 'sat_realizable', 'sat_synth_time', 'sat_accuracy',
                'sat_tp', 'sat_tn', 'sat_fp', 'sat_fn']})

        # ==================== samples2LTL[DT] ====================
        traces_dt = ExperimentTraces(
            tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
            operators=['G', 'F', '!', 'U', '&', '|', '->', 'X']
        )

        dt_time, dt_atoms, dt_prims = mine_dt(traces_dt)

        row['dt_time'] = dt_time
        row['dt_atoms'] = dt_atoms
        row['dt_prims'] = dt_prims
        row['dt_timeout'] = dt_time is None

        if dt_time is not None:
            print("  DT:       %.1fs  atoms=%s  prims=%s" % (dt_time, dt_atoms, dt_prims))
            # DT doesn't produce a single evaluatable formula — it produces a
            # decision tree over LTL atoms. Synthesis/generalization eval would
            # require reconstructing the tree formula, which is lossy.
            row['dt_realizable'] = None
            row['dt_synth_time'] = None
            row['dt_accuracy'] = None
        else:
            print("  DT:       FAILED")
            row['dt_realizable'] = None
            row['dt_synth_time'] = None
            row['dt_accuracy'] = None

        results.append(row)

    # ==================== Summary ====================
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    total = len(results)
    gr1_solved = sum(1 for r in results if not r['gr1_timeout'])
    sat_solved = sum(1 for r in results if not r['sat_timeout'])
    dt_solved = sum(1 for r in results if not r['dt_timeout'])
    gr1_real = sum(1 for r in results if r.get('gr1_realizable') is True)
    gr1_unreal = sum(1 for r in results if r.get('gr1_realizable') is False)
    sat_real = sum(1 for r in results if r.get('sat_realizable') is True)
    sat_unreal = sum(1 for r in results if r.get('sat_realizable') is False)

    print("Mined:        GR1 %d/%d    SAT %d/%d    DT %d/%d" % (
        gr1_solved, total, sat_solved, total, dt_solved, total))
    print("Realizable:   GR1 %d/%d    SAT %d/%d" % (
        gr1_real, gr1_solved, sat_real, sat_solved))
    print("Unrealizable: GR1 %d/%d    SAT %d/%d" % (
        gr1_unreal, gr1_solved, sat_unreal, sat_solved))

    gr1_accs = [r['gr1_accuracy'] for r in results if r.get('gr1_accuracy') is not None]
    sat_accs = [r['sat_accuracy'] for r in results if r.get('sat_accuracy') is not None]
    if gr1_accs:
        print("Avg gen acc:  GR1 %.1f%%" % (sum(gr1_accs) / len(gr1_accs) * 100))
    if sat_accs:
        print("              SAT %.1f%%" % (sum(sat_accs) / len(sat_accs) * 100))

    # Compact table
    print()
    hdr = "%-42s | %4s | %8s %5s %5s %5s | %8s %5s %5s %5s | %8s" % (
        'Benchmark', 'Vars',
        'GR1_t', 'Real', 'D', 'Acc',
        'SAT_t', 'Real', 'D', 'Acc',
        'DT_t')
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        def fmt_t(t):
            return ('%.1fs' % t) if t is not None else 'T/O'
        def fmt_r(v):
            if v is True: return 'Y'
            if v is False: return 'N'
            return '-'
        def fmt_a(v):
            return ('%.0f%%' % (v * 100)) if v is not None else '-'

        print("%-42s | %4d | %8s %5s %5s %5s | %8s %5s %5s %5s | %8s" % (
            r['benchmark'][:42], r['num_vars'],
            fmt_t(r['gr1_time']), fmt_r(r.get('gr1_realizable')),
            str(r['gr1_depth']) if r.get('gr1_depth') else '-',
            fmt_a(r.get('gr1_accuracy')),
            fmt_t(r['sat_time']), fmt_r(r.get('sat_realizable')),
            str(r.get('sat_depth')) if r.get('sat_depth') else '-',
            fmt_a(r.get('sat_accuracy')),
            fmt_t(r.get('dt_time'))))

    # Write CSV
    if results:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'full_eval_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\nResults saved to %s" % os.path.abspath(csv_path))


if __name__ == '__main__':
    main()
