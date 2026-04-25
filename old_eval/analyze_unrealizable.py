"""
Deep analysis of why GR1Mine specs are unrealizable.

For selected benchmarks:
1. Show ground truth vs mined spec side-by-side
2. Run slugs --counterStrategy to get env counter-strategy
3. Run slugs --analyzeAssumptions to check assumption issues
4. Check if assumptions are vacuously unsatisfiable
"""
import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.spectra_parser import parse_spectra_file
from utils.Traces import Trace, ExperimentTraces
from utils.trace_generator import generate_traces_sat
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from experiments.gr1_experiment import generate_template_configs, verify_formula
from experiments.synthesis_comparison import gr1_to_slugs, run_slugs_synthesis
from experiments.full_eval import generate_mixed_traces, mine_gr1
from z3 import sat as Z3_SAT

SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')
SLUGS_BIN = os.path.join(os.path.dirname(__file__), '..', '..', 'slugs', 'src', 'slugs')

# Pick a mix of benchmarks — small, medium, trivial
BENCHMARKS = [
    'cimattiAnalyzing/amba_ahb_w_guar_trans_amba_ahb_1.spectra',   # 16 vars, solved D=2
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_3.spectra', # 28 vars, solved D=1
    'cimattiAnalyzing/amba_ahb_w_guar_trans_amba_ahb_3.spectra',   # 28 vars, solved D=2
    'bloemDebugging/specs_G5wst2.spectra',                          # 24 vars, solved D=2
    'bloemDebugging/specs_G5wsf1.spectra',                          # 24 vars, solved D=2
    'cimattiAnalyzing/gen_buf_w_guar_fairness_5_genbuf.spectra',   # 24 vars, solved D=1
    'cimattiAnalyzing/gen_buf_w_guar_trans_5_genbuf.spectra',      # 24 vars, solved D=1
    'bloemDebugging/specs_A2wsf1.spectra',                          # 22 vars, solved D=3
]


def run_slugs_verbose(slugs_input, options=None):
    """Run slugs with extra options and return full output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slugsin', delete=False) as f:
        f.write(slugs_input)
        f.flush()
        tmppath = f.name

    try:
        cmd = [SLUGS_BIN]
        if options:
            cmd += options
        cmd.append(tmppath)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', ''
    finally:
        os.unlink(tmppath)


def check_assumption_satisfiability(gr1_formula, num_vars):
    """Check if the mined assumptions are even satisfiable by any trace."""
    from z3 import Solver, Bool, And, Or, Not, sat
    import itertools

    # Try to find a lasso trace (length 4, lasso at 2) that satisfies assumptions
    length = 5
    lasso = 2

    vars_at = [[Bool("v_%d_%d" % (t, i)) for i in range(num_vars)]
               for t in range(length)]

    s = Solver()

    # Encode assumptions
    from utils.trace_generator import _encode_assumptions
    constraints = _encode_assumptions(gr1_formula, vars_at, length, lasso)
    for c in constraints:
        s.add(c)

    if s.check() == sat:
        return True, s.model()
    return False, None


def check_guarantee_satisfiability(gr1_formula, num_vars):
    """Check if guarantees are satisfiable given assumptions."""
    from z3 import Solver, Bool, And, Or, Not, sat
    from utils.trace_generator import _encode_assumptions, _encode_guarantees

    length = 5
    lasso = 2
    vars_at = [[Bool("v_%d_%d" % (t, i)) for i in range(num_vars)]
               for t in range(length)]

    s = Solver()

    # Encode both assumptions AND guarantees
    for c in _encode_assumptions(gr1_formula, vars_at, length, lasso):
        s.add(c)
    for c in _encode_guarantees(gr1_formula, vars_at, length, lasso):
        s.add(c)

    if s.check() == sat:
        return True
    return False


def main():
    print("=" * 120)
    print("Analysis: Why are GR1Mine specs unrealizable?")
    print("=" * 120)

    for bm_path in BENCHMARKS:
        full_path = os.path.join(SPECTRA_DIR, bm_path)
        if not os.path.exists(full_path):
            print("SKIP %s — not found" % bm_path)
            continue

        try:
            gt_spec, meta = parse_spectra_file(full_path)
        except Exception as e:
            print("SKIP %s — parse error: %s" % (bm_path, e))
            continue

        name = os.path.basename(bm_path).replace('.spectra', '')
        num_vars = meta['num_vars']
        env_vars = meta['env_vars']
        sys_vars = meta['sys_vars']
        all_vars = meta['all_vars']
        env_indices = [meta['name_to_index'][v] for v in env_vars]

        print("\n" + "=" * 120)
        print("Benchmark: %s  (%d vars: %d env, %d sys)" % (
            name, num_vars, len(env_vars), len(sys_vars)))
        print("=" * 120)

        # Show ground truth
        print("\n--- GROUND TRUTH ---")
        if gt_spec.init_e:
            print("  init_e:   %s" % gt_spec.init_e.prettyPrint(True))
        if gt_spec.init_s:
            print("  init_s:   %s" % gt_spec.init_s.prettyPrint(True))
        if gt_spec.safety_e:
            print("  safety_e: %s" % gt_spec.safety_e.prettyPrint(True)[:100])
        if gt_spec.safety_s:
            print("  safety_s: %s" % gt_spec.safety_s.prettyPrint(True)[:100])
        for i, j in enumerate(gt_spec.justices):
            print("  J_%d:      %s" % (i, j.prettyPrint(True)))
        for i, g in enumerate(gt_spec.guarantees):
            print("  G_%d:      %s" % (i, g.prettyPrint(True)))

        # Verify ground truth is realizable
        gt_slugs = gr1_to_slugs(gt_spec, env_vars, sys_vars, all_vars)
        gt_out, gt_err = run_slugs_verbose(gt_slugs)
        gt_real = 'realizable' in gt_out.lower() and 'unrealizable' not in gt_out.lower()
        print("\n  Ground truth realizable: %s" % ('YES' if gt_real else 'NO'))

        # Mine
        print("\n--- MINING ---")
        train_pos, train_neg = generate_mixed_traces(
            gt_spec, num_vars, 25, 25, seed=hash(name) % 10000)

        if len(train_neg) < 3:
            print("  SKIP — only %d neg traces" % len(train_neg))
            continue

        traces = ExperimentTraces(
            tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
            operators=['&', '|', '!']
        )
        nj = len(gt_spec.justices)
        ng = len(gt_spec.guarantees)

        mined_f, mined_d, mined_t, mined_cfg = mine_gr1(
            traces, 5, min(nj + 1, 4), min(ng + 1, 4), env_indices, timeout=120)

        if not mined_f:
            print("  Mining TIMEOUT")
            continue

        print("  Mined (D=%s, %.1fs, config=%s):" % (mined_d, mined_t, mined_cfg))
        if mined_f.init_e:
            print("    init_e:   %s" % mined_f.init_e.prettyPrint(True))
        if mined_f.init_s:
            print("    init_s:   %s" % mined_f.init_s.prettyPrint(True))
        if mined_f.safety_e:
            print("    safety_e: %s" % mined_f.safety_e.prettyPrint(True))
        if mined_f.safety_s:
            print("    safety_s: %s" % mined_f.safety_s.prettyPrint(True))
        for i, j in enumerate(mined_f.justices):
            print("    J_%d:      %s" % (i, j.prettyPrint(True)))
        for i, g in enumerate(mined_f.guarantees):
            print("    G_%d:      %s" % (i, g.prettyPrint(True)))

        # Map variable indices to names for readability
        def render_var(label):
            if label.startswith('x') and label[-1].isdigit():
                primed = label.endswith("'")
                idx_str = label.rstrip("'")[1:]
                idx = int(idx_str)
                vname = all_vars[idx] if idx < len(all_vars) else label
                return vname + ("'" if primed else "")
            return label

        def render_formula(f):
            if f is None:
                return "None"
            s = f.prettyPrint(True)
            import re
            def repl(m):
                return render_var(m.group(0))
            return re.sub(r"x\d+'?", repl, s)

        print("\n  Mined (with variable names):")
        if mined_f.init_e:
            print("    init_e:   %s" % render_formula(mined_f.init_e))
        if mined_f.init_s:
            print("    init_s:   %s" % render_formula(mined_f.init_s))
        if mined_f.safety_e:
            print("    safety_e: %s" % render_formula(mined_f.safety_e))
        if mined_f.safety_s:
            print("    safety_s: %s" % render_formula(mined_f.safety_s))
        for i, j in enumerate(mined_f.justices):
            print("    J_%d:      %s" % (i, render_formula(j)))
        for i, g in enumerate(mined_f.guarantees):
            print("    G_%d:      %s" % (i, render_formula(g)))

        # Check assumption/guarantee satisfiability
        print("\n--- SATISFIABILITY ANALYSIS ---")
        assum_sat, _ = check_assumption_satisfiability(mined_f, num_vars)
        print("  Assumptions satisfiable: %s" % ('YES' if assum_sat else 'NO'))

        guar_sat = check_guarantee_satisfiability(mined_f, num_vars)
        print("  Assumptions + Guarantees jointly satisfiable: %s" % ('YES' if guar_sat else 'NO'))

        # Run slugs analysis
        print("\n--- SLUGS ANALYSIS ---")
        mined_slugs = gr1_to_slugs(mined_f, env_vars, sys_vars, all_vars)
        print("  Slugs input:")
        for line in mined_slugs.split('\n'):
            print("    %s" % line)

        # Basic realizability
        out, err = run_slugs_verbose(mined_slugs)
        print("\n  Realizability: %s" % out.strip())

        # Counter-strategy
        out_cs, err_cs = run_slugs_verbose(mined_slugs, ['--counterStrategy'])
        print("\n  Counter-strategy output:")
        for line in out_cs.strip().split('\n')[:30]:
            print("    %s" % line)

        # Analyze assumptions
        out_aa, err_aa = run_slugs_verbose(mined_slugs, ['--analyzeAssumptions'])
        print("\n  Assumption analysis:")
        for line in out_aa.strip().split('\n')[:20]:
            print("    %s" % line)
        if err_aa.strip():
            for line in err_aa.strip().split('\n')[:10]:
                print("    [stderr] %s" % line)

        print()

    print("\n" + "=" * 120)
    print("Done.")


if __name__ == '__main__':
    main()
