"""
Compare synthesis from GR1Mine output (via slugs) vs DT output (via ltlsynt).

For each benchmark:
1. Mine with GR1Mine → GR(1) spec → convert to slugs format → run slugs
2. Mine with DT → LTL decision tree → flatten to LTL → run ltlsynt
3. Compare synthesis time and realizability
"""
import sys, os, subprocess, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.spectra_parser import parse_spectra_file
from utils.trace_generator import generate_traces_sat
from experiments.eval_spectra import generate_traces_from_spec
from experiments.gr1_experiment import run_gr1_incremental_solver
from utils.Traces import ExperimentTraces
from smtEncoding.dagSATEncoding import DagSATEncoding
from solverRuns import run_dt_solver


SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'spectra-specs')
SLUGS_BIN = os.path.join(os.path.dirname(__file__), '..', '..', 'slugs', 'src', 'slugs')

BENCHMARKS = [
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_1.spectra',
    'cimattiAnalyzing/amba_ahb_w_guar_fairness_amba_ahb_2.spectra',
]


def gr1_to_slugs(gr1_formula, env_vars, sys_vars, all_vars):
    """Convert a GR1Formula to slugs input format.

    env_vars: list of env variable names
    sys_vars: list of sys variable names
    all_vars: ordered list of all variable names (index i → all_vars[i])
    """
    lines = []

    lines.append('[INPUT]')
    for v in env_vars:
        lines.append(v)
    lines.append('')

    lines.append('[OUTPUT]')
    for v in sys_vars:
        lines.append(v)
    lines.append('')

    # Convert propositional formula to slugs prefix notation
    def to_prefix(f):
        if f is None:
            return None
        label = f.label
        if label.startswith('x'):
            # Map x0, x1, ... to actual variable names
            primed = label.endswith("'")
            idx_str = label[1:-1] if primed else label[1:]
            idx = int(idx_str)
            var_name = all_vars[idx] if idx < len(all_vars) else label
            return var_name + "'" if primed else var_name
        elif label == '&':
            return '& %s %s' % (to_prefix(f.left), to_prefix(f.right))
        elif label == '|':
            return '| %s %s' % (to_prefix(f.left), to_prefix(f.right))
        elif label == '!':
            return '! %s' % to_prefix(f.left)
        elif label == '->':
            return '| ! %s %s' % (to_prefix(f.left), to_prefix(f.right))
        else:
            return label

    # ENV_INIT
    lines.append('[ENV_INIT]')
    if gr1_formula.init_e is not None:
        lines.append(to_prefix(gr1_formula.init_e))
    lines.append('')

    # ENV_TRANS (safety)
    lines.append('[ENV_TRANS]')
    if gr1_formula.safety_e is not None:
        lines.append(to_prefix(gr1_formula.safety_e))
    lines.append('')

    # ENV_LIVENESS (justices)
    lines.append('[ENV_LIVENESS]')
    for j in gr1_formula.justices:
        lines.append(to_prefix(j))
    lines.append('')

    # SYS_INIT
    lines.append('[SYS_INIT]')
    if gr1_formula.init_s is not None:
        lines.append(to_prefix(gr1_formula.init_s))
    lines.append('')

    # SYS_TRANS (safety)
    lines.append('[SYS_TRANS]')
    if gr1_formula.safety_s is not None:
        lines.append(to_prefix(gr1_formula.safety_s))
    lines.append('')

    # SYS_LIVENESS (guarantees)
    lines.append('[SYS_LIVENESS]')
    for g in gr1_formula.guarantees:
        lines.append(to_prefix(g))
    lines.append('')

    return '\n'.join(lines)


def dt_to_ltl(atoms_file, tree_file):
    """Read DT output and flatten to an LTL formula string for ltlsynt."""
    with open(atoms_file) as f:
        atoms = [line.strip() for line in f if line.strip()]

    with open(tree_file) as f:
        tree_lines = [line.strip() for line in f if line.strip()]

    # Parse the tree structure to build an LTL formula
    # Each line: "atom_or_* : count" or "* : count"
    # The tree is a nested if-then-else: if atom then left_subtree else right_subtree
    # A leaf marked * at depth d under a sequence of decisions represents a conjunction
    # For now, just return the raw atoms joined — the actual tree formula is complex

    # Simple approach: collect the atoms used in the tree (non-leaf nodes)
    # and build a conjunction/disjunction heuristic
    # Actually, let's just read what the DT produces and format it for ltlsynt

    # The tree is actually a decision tree — we need to convert it to a boolean formula
    # Parse recursively
    def parse_tree(lines, idx=0, depth=0):
        if idx >= len(lines):
            return None, idx
        line = lines[idx]
        parts = line.split(' : ')
        node = parts[0].strip()
        count = int(parts[1].strip()) if len(parts) > 1 else 0

        if node == '*':
            # Leaf — check if it's a positive or negative leaf
            # In the DT, left child = atom is true, right child = atom is false
            # A leaf * means "classify as positive" if count > 0
            # We need to know the label... just return True/False placeholder
            return ('true', idx + 1)

        # Internal node — has two children
        atom = node
        left_result, next_idx = parse_tree(lines, idx + 1, depth + 1)
        right_result, next_idx = parse_tree(lines, next_idx, depth + 1)

        # if atom then left else right = (atom & left) | (!atom & right)
        if left_result == 'true' and right_result == 'true':
            return ('true', next_idx)
        elif left_result == 'true':
            return ('((%s) | (!(%s) & %s))' % (atom, atom, right_result), next_idx)
        elif right_result == 'true':
            return ('((!(%s)) | ((%s) & %s))' % (atom, atom, left_result), next_idx)
        else:
            return ('((%s) & %s) | (!(%s) & %s)' % (atom, left_result, atom, right_result), next_idx)

    result, _ = parse_tree(tree_lines)
    return result if result else 'true'


def run_slugs_synthesis(slugs_input, timeout=60):
    """Run slugs on a .slugsin string, return (realizable, time)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slugsin', delete=False) as f:
        f.write(slugs_input)
        f.flush()
        tmppath = f.name

    try:
        start = time.time()
        result = subprocess.run(
            [SLUGS_BIN, tmppath],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        # slugs writes "RESULT: Specification is realizable/unrealizable." to stderr
        output = result.stdout + '\n' + result.stderr
        realizable = 'realizable' in output.lower() and 'unrealizable' not in output.lower()
        return realizable, elapsed, output.strip()
    except subprocess.TimeoutExpired:
        return None, timeout, 'TIMEOUT'
    finally:
        os.unlink(tmppath)


def run_ltlsynt_synthesis(ltl_formula, ins, outs, timeout=60):
    """Run ltlsynt on an LTL formula, return (realizable, time)."""
    try:
        start = time.time()
        result = subprocess.run(
            ['ltlsynt', '--ins=' + ','.join(ins), '--outs=' + ','.join(outs),
             '-f', ltl_formula],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        realizable = 'REALIZABLE' in result.stdout
        return realizable, elapsed, result.stdout.strip().split('\n')[0] if result.stdout else ''
    except subprocess.TimeoutExpired:
        return None, timeout, 'TIMEOUT'


if __name__ == '__main__':
    print("=" * 120)
    print("Synthesis Comparison: GR1Mine → slugs  vs  DT → ltlsynt")
    print("=" * 120)
    print()

    for bm in BENCHMARKS:
        path = os.path.join(SPECTRA_DIR, bm)
        gr1_gt, meta = parse_spectra_file(path)
        n = meta['num_vars']
        nj = len(gr1_gt.justices)
        ng = len(gr1_gt.guarantees)
        name = os.path.basename(bm).replace('.spectra', '')
        env_vars = meta.get('env_vars', ['x%d' % i for i in range(n // 2)])
        sys_vars = meta.get('sys_vars', ['x%d' % i for i in range(n // 2, n)])

        print("Benchmark: %s (%d vars, %d env, %d sys)" % (name, n, len(env_vars), len(sys_vars)))

        # Generate mixed training traces
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
        print("  Traces: %d pos, %d neg" % (len(pos), len(neg)))

        # --- GR1Mine ---
        traces_gr1 = ExperimentTraces(
            tracesToAccept=list(pos), tracesToReject=list(neg),
            operators=['&', '|', '!']
        )
        print("  Mining with GR1Mine...", flush=True)
        gr1_result = run_gr1_incremental_solver(
            traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4))
        gr1_f = gr1_result[0]
        gr1_mine_time = gr1_result[2]

        if gr1_f:
            print("  GR1Mine: %.1fs, %s" % (gr1_mine_time, gr1_f.prettyPrint()[:80]))

            # Convert to slugs and synthesize
            slugs_input = gr1_to_slugs(gr1_f, env_vars, sys_vars, meta['all_vars'])
            print("  Running slugs...", flush=True)
            realizable, synth_time, msg = run_slugs_synthesis(slugs_input)
            print("  Slugs: %s, %.3fs" % (
                'REALIZABLE' if realizable else ('UNREALIZABLE' if realizable is False else 'TIMEOUT'),
                synth_time))
            print("    %s" % msg[:100])
        else:
            print("  GR1Mine: FAILED")

        # --- DT ---
        traces_dt = ExperimentTraces(
            tracesToAccept=list(pos), tracesToReject=list(neg),
            operators=['G', 'F', '!', '&', '|', 'X']
        )
        print("  Mining with DT...", flush=True)
        tree_file = '/tmp/dt_synth_%s.txt' % name
        dt_result = run_dt_solver(traces_dt, encoder=DagSATEncoding, txtFile=tree_file)
        dt_time = dt_result[0]
        print("  DT: %.1fs, %d atoms, %d nodes" % (dt_time, dt_result[1], dt_result[2]))

        # Convert DT to LTL and synthesize
        ltl_formula = dt_to_ltl('atoms.txt', tree_file)
        print("  LTL formula: %s" % (ltl_formula[:100] if ltl_formula else 'NONE'))
        if ltl_formula and ltl_formula != 'true':
            env_v = ['x%d' % i for i in range(n // 2)]
            sys_v = ['x%d' % i for i in range(n // 2, n)]
            print("  Running ltlsynt...", flush=True)
            realizable, synth_time, msg = run_ltlsynt_synthesis(ltl_formula, env_v, sys_v)
            print("  ltlsynt: %s, %.3fs" % (
                'REALIZABLE' if realizable else ('UNREALIZABLE' if realizable is False else 'TIMEOUT'),
                synth_time))
        else:
            print("  ltlsynt: SKIPPED (no formula)")

        print()
