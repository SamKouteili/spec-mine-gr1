"""
Quick benchmark: partitioned vs unpartitioned GR(1) mining on Spectra specs.

Compares mining with env/sys variable partition constraint (for realizability)
against the original unconstrained mining.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random
from z3 import sat

from utils.Traces import Trace, ExperimentTraces
from utils.spectra_parser import parse_spectra_file
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from experiments.gr1_experiment import generate_template_configs, verify_formula


# Benchmarks to test
SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')

BENCHMARKS = [
    'bloemDebugging/specs_A2wst1.spectra',
    'bloemDebugging/specs_A2wst2.spectra',
    'bloemDebugging/specs_A4wst1.spectra',
    'bloemDebugging/specs_A4wsf1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_2.spectra',
]

TIMEOUT = 120  # 2 minutes per run
MAX_DEPTH = 5
MAX_JUSTICES = 3
MAX_GUARANTEES = 3


def generate_traces(spec, num_vars, num_pos=15, num_neg=10,
                    trace_length=5, max_trials=50000, seed=42):
    """Generate traces from a GR(1) spec (same as eval_spectra)."""
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


def run_mining(traces, max_depth, max_justices, max_guarantees, env_var_indices=None):
    """Run GR(1) miner with template enumeration. Returns (formula, D, time, config)."""
    start = time.time()
    configs = generate_template_configs(max_justices, max_guarantees)

    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            if time.time() - start > TIMEOUT:
                return None, None, TIMEOUT, None

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
                    return formula, D, elapsed, (include_init, include_safety, m, k)

    elapsed = time.time() - start
    return None, None, elapsed, None


def check_variable_placement(formula, env_indices, sys_indices, num_vars):
    """Check which variables appear on assumption vs guarantee side."""
    def collect_vars(f):
        if f is None:
            return set()
        if f.label.startswith('x'):
            name = f.label.rstrip("'")
            idx = int(name[1:])
            return {idx}
        vars_found = set()
        if f.left:
            vars_found |= collect_vars(f.left)
        if f.right:
            vars_found |= collect_vars(f.right)
        return vars_found

    env_set = set(env_indices)
    sys_set = set(sys_indices)

    # Check justice assumptions
    issues = []
    for i, j in enumerate(formula.justices):
        vars_in = collect_vars(j)
        sys_in_justice = vars_in & sys_set
        if sys_in_justice:
            issues.append("  Justice J_%d has sys vars: {%s}" % (
                i, ', '.join('x%d' % v for v in sorted(sys_in_justice))))

    # Check guarantee
    for i, g in enumerate(formula.guarantees):
        vars_in = collect_vars(g)
        env_in_guar = vars_in & env_set
        if env_in_guar:
            issues.append("  Guarantee G_%d has env vars: {%s}" % (
                i, ', '.join('x%d' % v for v in sorted(env_in_guar))))

    return issues


def main():
    print("=" * 100)
    print("Partitioned vs Unpartitioned GR(1) Mining")
    print("=" * 100)
    print()

    for bm_path in BENCHMARKS:
        full_path = os.path.join(SPECTRA_DIR, bm_path)
        if not os.path.exists(full_path):
            print("SKIP %s — file not found" % bm_path)
            continue

        try:
            spec, meta = parse_spectra_file(full_path)
        except Exception as e:
            print("SKIP %s — parse error: %s" % (bm_path, e))
            continue

        name = os.path.basename(bm_path).replace('.spectra', '')
        num_vars = meta['num_vars']
        env_indices = [meta['name_to_index'][v] for v in meta['env_vars']]
        sys_indices = [meta['name_to_index'][v] for v in meta['sys_vars']]

        print("-" * 100)
        print("Benchmark: %s  |  vars: %d (env: %d, sys: %d)  |  J: %d, G: %d" % (
            name, num_vars, len(env_indices), len(sys_indices),
            len(spec.justices), len(spec.guarantees)))
        print("  env vars: %s" % meta['env_vars'])
        print("  sys vars: %s" % meta['sys_vars'])

        # Generate traces
        pos, neg = generate_traces(spec, num_vars, seed=hash(name) % 10000)
        if len(neg) < 3:
            print("  SKIP — only %d negative traces" % len(neg))
            continue
        print("  traces: %d pos, %d neg" % (len(pos), len(neg)))

        traces = ExperimentTraces(
            tracesToAccept=pos, tracesToReject=neg,
            operators=['&', '|', '!']
        )

        # --- Unpartitioned ---
        f_unpart, d_unpart, t_unpart, cfg_unpart = run_mining(
            traces, MAX_DEPTH, MAX_JUSTICES, MAX_GUARANTEES,
            env_var_indices=None)

        if f_unpart:
            issues_unpart = check_variable_placement(f_unpart, env_indices, sys_indices, num_vars)
            print("\n  [UNPARTITIONED] D=%s, time=%.3fs, config=%s" % (d_unpart, t_unpart, cfg_unpart))
            print("    Formula: %s" % f_unpart.prettyPrint())
            if issues_unpart:
                print("    Variable placement issues:")
                for iss in issues_unpart:
                    print("    %s" % iss)
            else:
                print("    Variable placement: OK")
        else:
            print("\n  [UNPARTITIONED] UNSAT/TIMEOUT (%.1fs)" % t_unpart)

        # --- Partitioned ---
        f_part, d_part, t_part, cfg_part = run_mining(
            traces, MAX_DEPTH, MAX_JUSTICES, MAX_GUARANTEES,
            env_var_indices=env_indices)

        if f_part:
            issues_part = check_variable_placement(f_part, env_indices, sys_indices, num_vars)
            print("\n  [PARTITIONED]   D=%s, time=%.3fs, config=%s" % (d_part, t_part, cfg_part))
            print("    Formula: %s" % f_part.prettyPrint())
            if issues_part:
                print("    Variable placement issues:")
                for iss in issues_part:
                    print("    %s" % iss)
            else:
                print("    Variable placement: OK")
        else:
            print("\n  [PARTITIONED]   UNSAT/TIMEOUT (%.1fs)" % t_part)

        print()

    print("=" * 100)
    print("Done.")


if __name__ == '__main__':
    main()
