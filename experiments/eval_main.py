"""
Main evaluation: GR1Mine vs ATLAS[LTL] vs ATLAS[GR1] on Spectra benchmarks.

For each benchmark:
1. Parse ground-truth GR(1) spec, generate 25+25 mixed traces
2. Mine with GR1Mine (partitioned), ATLAS unconstrained, ATLAS GR(1)-constrained
3. Synthesize: GR1Mine + ATLAS[GR1] via slugs, ATLAS[LTL] via ltlsynt
4. Report time, formula, realizability
"""
import sys, os, subprocess, tempfile, time, random, csv, glob as globmod
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from z3 import sat as Z3_SAT
from utils.spectra_parser import parse_spectra_file
from utils.Traces import Trace, ExperimentTraces
from utils.trace_generator import generate_traces_sat
from utils.GR1Formula import GR1Formula
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from experiments.gr1_experiment import generate_template_configs, verify_formula

SPECTRA_DIR = os.path.join(os.path.dirname(__file__), '..', 'benchmarks')
SLUGS_BIN = os.path.join(os.path.dirname(__file__), '..', '..', 'slugs', 'src', 'slugs')
ATLAS_JAR = os.path.join(os.path.dirname(__file__), '..', '..', 'ATLAS', 'bin', 'Atlas.jar')

MINE_TIMEOUT = 300
SYNTH_TIMEOUT = 60
TRAIN_POS = 25
TRAIN_NEG = 25


# ============================================================
# Trace generation
# ============================================================

def generate_random_traces(spec, num_vars, num_pos, num_neg,
                           trace_length=5, max_trials=50000, seed=42):
    random.seed(seed)
    neg_ah, neg_af, pos_g, pos_v = [], [], [], []
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
            (pos_g if assum else pos_v).append(tr)
        else:
            tr.intendedEvaluation = False
            (neg_ah if assum else neg_af).append(tr)
        if len(neg_ah) >= num_neg * 2 and len(pos_g) >= num_pos and len(pos_v) >= num_pos:
            break
    sel_neg = neg_ah[:num_neg]
    if len(sel_neg) < num_neg:
        sel_neg += neg_af[:num_neg - len(sel_neg)]
    half = num_pos // 2
    sel_pos = pos_g[:half] + pos_v[:half]
    remaining = num_pos - len(sel_pos)
    if remaining > 0:
        sel_pos += (pos_g[half:] + pos_v[half:])[:remaining]
    return sel_pos, sel_neg


def generate_mixed_traces(spec, num_vars, num_pos, num_neg, seed=42):
    half_pos, half_neg = num_pos // 2, num_neg // 2
    sat_pos, sat_neg = generate_traces_sat(
        spec, num_vars, num_pos=half_pos, num_neg=half_neg,
        trace_lengths=[4, 5, 6], max_trials_per_config=20)
    rand_pos, rand_neg = generate_random_traces(
        spec, num_vars, num_pos - len(sat_pos), num_neg - len(sat_neg), seed=seed)
    return (sat_pos + rand_pos)[:num_pos], (sat_neg + rand_neg)[:num_neg]


# ============================================================
# Trace I/O for ATLAS
# ============================================================

def traces_to_atlas_format(pos_traces, neg_traces):
    """Convert traces to ATLAS .trace file format (without header sections)."""
    def trace_str(tr):
        states = [','.join(str(int(v)) for v in step) for step in tr.traceVector]
        return ';'.join(states) + '::' + str(tr.lassoStart)
    pos_lines = [trace_str(tr) for tr in pos_traces]
    neg_lines = [trace_str(tr) for tr in neg_traces]
    return '\n'.join(pos_lines), '\n'.join(neg_lines)


def write_atlas_unconstrained(pos_str, neg_str, num_vars, max_depth, path):
    """Write ATLAS trace file for unconstrained LTL mining."""
    with open(path, 'w') as f:
        f.write(pos_str + '\n---\n' + neg_str + '\n---\n')
        f.write('G,F,!,U,&,|,->,X\n---\n')
        f.write('%d\n---\n\n' % max_depth)


def write_atlas_gr1(pos_str, neg_str, num_vars, max_depth, env_indices, path):
    """Write ATLAS trace file with GR(1) + env/sys partition constraint."""
    env_lits = ' + '.join(['x%d' % i for i in env_indices])
    sys_lits = ' + '.join(['x%d' % i for i in range(num_vars) if i not in set(env_indices)])

    with open(path, 'w') as f:
        f.write(pos_str + '\n---\n' + neg_str + '\n---\n')
        f.write('G,F,!,U,&,|,->,X\n---\n')
        f.write('%d\n---\n\n---\n' % max_depth)
        f.write('fact {\n')
        f.write('    root in Imply\n')
        f.write('    root.l in G\n')
        f.write('    root.l.l in F\n')
        f.write('    root.r in G\n')
        f.write('    root.r.l in F\n')
        f.write('    all n: root.l.l.l.*(l+r) | n in And + Or + Neg + Literal\n')
        f.write('    all n: root.r.l.l.*(l+r) | n in And + Or + Neg + Literal\n')
        f.write('    all n: root.l.l.l.*(l+r) & Literal | n in %s\n' % env_lits)
        f.write('    all n: root.r.l.l.*(l+r) & Literal | n in %s\n' % sys_lits)
        f.write('}\n')


# ============================================================
# Mining
# ============================================================

def mine_gr1(traces, max_depth, max_justices, max_guarantees,
             env_var_indices, timeout=MINE_TIMEOUT):
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


def mine_atlas(trace_file, timeout=MINE_TIMEOUT):
    """Run ATLAS on a trace file. Returns (formula_str, time, raw_output)."""
    try:
        result = subprocess.run(
            ['java', '-cp', ATLAS_JAR,
             'cmu.s3d.ltl.app.CLIKt',
             '-f', trace_file,
             '-T', str(timeout)],
            capture_output=True, text=True, timeout=timeout + 30)
        output = result.stdout.strip()
        # Parse CSV output using csv module to handle quoted fields
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return None, None, output
        import io
        reader = csv.reader(io.StringIO(lines[-1]))
        fields = list(reader)[0]
        # Fields: filename, nPos, nNeg, maxOps, nVars, maxLen, expected, solvingTime, formula
        solve_time_str = fields[7]
        formula_str = fields[8] if len(fields) > 8 else '-'
        if solve_time_str == 'TO' or formula_str in ('TO', 'UNSAT', '-'):
            return None, timeout, output
        solve_time = float(solve_time_str)
        return formula_str, solve_time, output
    except subprocess.TimeoutExpired:
        return None, timeout, 'TIMEOUT'
    except Exception as e:
        return None, None, 'ERR: %s' % str(e)


# ============================================================
# Synthesis
# ============================================================

def gr1_to_slugs(gr1_formula, env_vars, sys_vars, all_vars):
    """Convert a GR1Formula to slugs input format."""
    lines = []
    def to_prefix(f):
        if f is None:
            return None
        label = f.label
        if label.startswith('x'):
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

    lines.append('[INPUT]')
    for v in env_vars:
        lines.append(v)
    lines.append('')
    lines.append('[OUTPUT]')
    for v in sys_vars:
        lines.append(v)
    lines.append('')
    lines.append('[ENV_INIT]')
    if gr1_formula.init_e is not None:
        lines.append(to_prefix(gr1_formula.init_e))
    lines.append('')
    lines.append('[ENV_TRANS]')
    if gr1_formula.safety_e is not None:
        lines.append(to_prefix(gr1_formula.safety_e))
    lines.append('')
    lines.append('[ENV_LIVENESS]')
    for j in gr1_formula.justices:
        lines.append(to_prefix(j))
    lines.append('')
    lines.append('[SYS_INIT]')
    if gr1_formula.init_s is not None:
        lines.append(to_prefix(gr1_formula.init_s))
    lines.append('')
    lines.append('[SYS_TRANS]')
    if gr1_formula.safety_s is not None:
        lines.append(to_prefix(gr1_formula.safety_s))
    lines.append('')
    lines.append('[SYS_LIVENESS]')
    for g in gr1_formula.guarantees:
        lines.append(to_prefix(g))
    lines.append('')
    return '\n'.join(lines)


def atlas_gr1_to_slugs(formula_str, env_vars, sys_vars):
    """Convert ATLAS GR(1) formula string to slugs input.

    ATLAS GR(1) formulas have the form: ->(G(F(lhs)),G(F(rhs)))
    We parse out the lhs and rhs propositional formulas.
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
    lines.append('[ENV_INIT]')
    lines.append('')
    lines.append('[ENV_TRANS]')
    lines.append('')

    # Parse: ->(G(F(lhs)),G(F(rhs)))
    # Strip outer ->( and trailing )
    all_vars = env_vars + sys_vars

    def varname(x_label):
        """Convert x0, x1, ... to actual variable name."""
        idx = int(x_label[1:])
        return all_vars[idx] if idx < len(all_vars) else x_label

    def to_slugs_prefix(s):
        """Convert ATLAS prefix notation to slugs prefix notation."""
        s = s.strip()
        if s.startswith('&('):
            inner = s[2:-1]
            a, b = _split_args(inner)
            return '& %s %s' % (to_slugs_prefix(a), to_slugs_prefix(b))
        elif s.startswith('|('):
            inner = s[2:-1]
            a, b = _split_args(inner)
            return '| %s %s' % (to_slugs_prefix(a), to_slugs_prefix(b))
        elif s.startswith('!('):
            inner = s[2:-1]
            return '! %s' % to_slugs_prefix(inner)
        elif s.startswith('x'):
            return varname(s)
        return s

    def _split_args(s):
        """Split 'a,b' at the top-level comma."""
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                return s[:i], s[i+1:]
        return s, ''

    # Parse the ATLAS output: ->(G(F(lhs)),G(F(rhs)))
    if formula_str.startswith('->(G(F('):
        inner = formula_str[3:-1]  # G(F(lhs)),G(F(rhs))
        lhs_gf, rhs_gf = _split_args(inner)
        # lhs_gf = G(F(lhs)), rhs_gf = G(F(rhs))
        lhs = lhs_gf[4:-2]  # strip G(F( and ))
        rhs = rhs_gf[4:-2]

        lines.append('[ENV_LIVENESS]')
        lines.append(to_slugs_prefix(lhs))
        lines.append('')
        lines.append('[SYS_INIT]')
        lines.append('')
        lines.append('[SYS_TRANS]')
        lines.append('')
        lines.append('[SYS_LIVENESS]')
        lines.append(to_slugs_prefix(rhs))
        lines.append('')
    else:
        # Can't parse — return None
        return None

    return '\n'.join(lines)


def synthesize_slugs(slugs_input):
    """Run slugs, return (realizable, time)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slugsin', delete=False) as f:
        f.write(slugs_input)
        f.flush()
        tmp = f.name
    try:
        start = time.time()
        result = subprocess.run([SLUGS_BIN, tmp],
                                capture_output=True, text=True, timeout=SYNTH_TIMEOUT)
        elapsed = time.time() - start
        output = result.stderr  # slugs writes to stderr!
        if 'unrealizable' in output.lower():
            return False, elapsed
        elif 'realizable' in output.lower():
            return True, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, SYNTH_TIMEOUT
    finally:
        os.unlink(tmp)


def synthesize_ltlsynt(formula_str, env_vars, sys_vars, all_vars):
    """Synthesize flat LTL formula via ltlsynt."""
    def to_ltlsynt(s, all_v):
        """Convert ATLAS prefix notation to ltlsynt infix."""
        s = s.strip()
        if s.startswith('->('):
            inner = s[3:-1]
            a, b = _split_args_ltl(inner)
            return '(%s -> %s)' % (to_ltlsynt(a, all_v), to_ltlsynt(b, all_v))
        elif s.startswith('G('):
            return 'G(%s)' % to_ltlsynt(s[2:-1], all_v)
        elif s.startswith('F('):
            return 'F(%s)' % to_ltlsynt(s[2:-1], all_v)
        elif s.startswith('X('):
            return 'X(%s)' % to_ltlsynt(s[2:-1], all_v)
        elif s.startswith('U('):
            inner = s[2:-1]
            a, b = _split_args_ltl(inner)
            return '(%s U %s)' % (to_ltlsynt(a, all_v), to_ltlsynt(b, all_v))
        elif s.startswith('&('):
            inner = s[2:-1]
            a, b = _split_args_ltl(inner)
            return '(%s & %s)' % (to_ltlsynt(a, all_v), to_ltlsynt(b, all_v))
        elif s.startswith('|('):
            inner = s[2:-1]
            a, b = _split_args_ltl(inner)
            return '(%s | %s)' % (to_ltlsynt(a, all_v), to_ltlsynt(b, all_v))
        elif s.startswith('!('):
            return '!(%s)' % to_ltlsynt(s[2:-1], all_v)
        elif s.startswith('x'):
            idx = int(s[1:])
            return all_v[idx].lower() if idx < len(all_v) else s
        return s

    def _split_args_ltl(s):
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                return s[:i], s[i+1:]
        return s, ''

    ltl_str = to_ltlsynt(formula_str, all_vars)
    env_lc = [v.lower() for v in env_vars]
    sys_lc = [v.lower() for v in sys_vars]
    try:
        start = time.time()
        result = subprocess.run(
            ['ltlsynt',
             '--ins=' + ','.join(env_lc),
             '--outs=' + ','.join(sys_lc),
             '-f', ltl_str],
            capture_output=True, text=True, timeout=SYNTH_TIMEOUT)
        elapsed = time.time() - start
        if 'REALIZABLE' in result.stdout and 'UNREALIZABLE' not in result.stdout:
            return True, elapsed
        elif 'UNREALIZABLE' in result.stdout:
            return False, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, SYNTH_TIMEOUT


# ============================================================
# Benchmark discovery
# ============================================================

def find_benchmarks(max_vars=30):
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
    # Sort by variable count (smallest first) for faster early results
    benchmarks.sort(key=lambda b: b['metadata']['num_vars'])
    return benchmarks


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-vars', type=int, default=300)
    parser.add_argument('--benchmarks', type=str, default=None,
                        help='Comma-separated benchmark names (substrings) to run')
    parser.add_argument('--miners', type=str, default=None,
                        help='Comma-separated miners to run: gr1,atlas_ltl,atlas_gr1 (default: all)')
    parser.add_argument('--append-csv', action='store_true',
                        help='Append results to existing CSV instead of overwriting')
    args = parser.parse_args()

    ALL_MINERS = {'gr1', 'atlas_ltl', 'atlas_gr1'}
    if args.miners:
        active_miners = {m.strip() for m in args.miners.split(',')}
        unknown = active_miners - ALL_MINERS
        if unknown:
            parser.error("Unknown miners: %s. Choose from: %s" % (
                ', '.join(unknown), ', '.join(sorted(ALL_MINERS))))
    else:
        active_miners = ALL_MINERS
    run_gr1 = 'gr1' in active_miners
    run_atlas_ltl = 'atlas_ltl' in active_miners
    run_atlas_gr1 = 'atlas_gr1' in active_miners
    benchmarks = find_benchmarks(max_vars=args.max_vars)
    if args.benchmarks:
        filters = [f.strip() for f in args.benchmarks.split(',')]
        benchmarks = [b for b in benchmarks
                      if any(f in b['name'] for f in filters)]
    miner_names = []
    if run_gr1: miner_names.append('GR1Mine')
    if run_atlas_ltl: miner_names.append('ATLAS[LTL]')
    if run_atlas_gr1: miner_names.append('ATLAS[GR1]')
    print("=" * 140)
    print("Evaluation: %s  (%d benchmarks)" % (' vs '.join(miner_names), len(benchmarks)))
    print("Traces: %d pos + %d neg (mixed SAT + random)" % (TRAIN_POS, TRAIN_NEG))
    print("=" * 140)

    results = []
    tmpdir = tempfile.mkdtemp(prefix='atlas_eval_')

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

        # Generate traces
        try:
            train_pos, train_neg = generate_mixed_traces(
                spec, num_vars, TRAIN_POS, TRAIN_NEG, seed=hash(name) % 10000)
        except (RecursionError, Exception) as e:
            print("  SKIP — trace generation failed: %s" % type(e).__name__)
            continue
        if len(train_neg) < 3:
            print("  SKIP — only %d negative traces" % len(train_neg))
            continue
        print("  Traces: %d pos, %d neg" % (len(train_pos), len(train_neg)))

        row = {
            'benchmark': name, 'num_vars': num_vars,
            'num_env': len(env_vars), 'num_sys': len(sys_vars),
            'nJ': nj, 'nG': ng,
            'train_pos': len(train_pos), 'train_neg': len(train_neg),
        }

        # Write ATLAS trace files (only if needed)
        if run_atlas_ltl or run_atlas_gr1:
            pos_str, neg_str = traces_to_atlas_format(train_pos, train_neg)
            atlas_uncon_path = os.path.join(tmpdir, '%s_uncon.trace' % name)
            atlas_gr1_path = os.path.join(tmpdir, '%s_gr1.trace' % name)
            if run_atlas_ltl:
                write_atlas_unconstrained(pos_str, neg_str, num_vars, 5, atlas_uncon_path)
            if run_atlas_gr1:
                write_atlas_gr1(pos_str, neg_str, num_vars, 5, env_indices, atlas_gr1_path)

        # ==================== GR1Mine ====================
        if run_gr1:
            traces_gr1 = ExperimentTraces(
                tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
                operators=['&', '|', '!'])
            gr1_f, gr1_d, gr1_time, gr1_cfg = mine_gr1(
                traces_gr1, 5, min(nj + 1, 4), min(ng + 1, 4), env_indices)

            row['gr1_time'] = gr1_time
            row['gr1_depth'] = gr1_d
            row['gr1_formula'] = str(gr1_f) if gr1_f else None
            row['gr1_timeout'] = gr1_f is None

            if gr1_f:
                print("  GR1Mine:     D=%s  %.1fs  %s" % (gr1_d, gr1_time, str(gr1_f)[:80]))
                slugs_in = gr1_to_slugs(gr1_f, env_vars, sys_vars, all_vars)
                real, synth_t = synthesize_slugs(slugs_in)
                row['gr1_realizable'] = real
                row['gr1_synth_time'] = synth_t
                status = 'REALIZABLE' if real else ('UNREALIZABLE' if real is False else 'ERROR')
                print("    Synthesis (slugs): %s (%.3fs)" % (status, synth_t))
            else:
                print("  GR1Mine:     TIMEOUT (%.1fs)" % gr1_time)
                row['gr1_realizable'] = None
                row['gr1_synth_time'] = None

        # ==================== ATLAS[LTL] ====================
        if run_atlas_ltl:
            atlas_ltl_f, atlas_ltl_t, atlas_ltl_raw = mine_atlas(atlas_uncon_path)

            row['atlas_ltl_time'] = atlas_ltl_t
            row['atlas_ltl_formula'] = atlas_ltl_f
            row['atlas_ltl_timeout'] = atlas_ltl_f is None

            if atlas_ltl_f:
                print("  ATLAS[LTL]:  %.1fs  %s" % (atlas_ltl_t, atlas_ltl_f[:80]))
                real, synth_t = synthesize_ltlsynt(atlas_ltl_f, env_vars, sys_vars, all_vars)
                row['atlas_ltl_realizable'] = real
                row['atlas_ltl_synth_time'] = synth_t
                status = 'REALIZABLE' if real else ('UNREALIZABLE' if real is False else 'ERROR')
                print("    Synthesis (ltlsynt): %s (%.3fs)" % (status, synth_t))
            else:
                print("  ATLAS[LTL]:  TIMEOUT (%.1fs)" % (atlas_ltl_t or 0))
                row['atlas_ltl_realizable'] = None
                row['atlas_ltl_synth_time'] = None

        # ==================== ATLAS[GR1] ====================
        if run_atlas_gr1:
            atlas_gr1_f, atlas_gr1_t, atlas_gr1_raw = mine_atlas(atlas_gr1_path)

            row['atlas_gr1_time'] = atlas_gr1_t
            row['atlas_gr1_formula'] = atlas_gr1_f
            row['atlas_gr1_timeout'] = atlas_gr1_f is None

            if atlas_gr1_f:
                print("  ATLAS[GR1]:  %.1fs  %s" % (atlas_gr1_t, atlas_gr1_f[:80]))
                slugs_in = atlas_gr1_to_slugs(atlas_gr1_f, env_vars, sys_vars)
                if slugs_in:
                    real, synth_t = synthesize_slugs(slugs_in)
                    row['atlas_gr1_realizable'] = real
                    row['atlas_gr1_synth_time'] = synth_t
                    status = 'REALIZABLE' if real else ('UNREALIZABLE' if real is False else 'ERROR')
                    print("    Synthesis (slugs): %s (%.3fs)" % (status, synth_t))
                else:
                    print("    Synthesis: SKIP (can't parse formula for slugs)")
                    row['atlas_gr1_realizable'] = None
                    row['atlas_gr1_synth_time'] = None
            else:
                print("  ATLAS[GR1]:  TIMEOUT (%.1fs)" % (atlas_gr1_t or 0))
                row['atlas_gr1_realizable'] = None
                row['atlas_gr1_synth_time'] = None

        results.append(row)
        sys.stdout.flush()

    # ==================== Summary ====================
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    total = len(results)
    gr1_solved = sum(1 for r in results if not r['gr1_timeout'])
    altl_solved = sum(1 for r in results if not r['atlas_ltl_timeout'])
    agr1_solved = sum(1 for r in results if not r['atlas_gr1_timeout'])
    gr1_real = sum(1 for r in results if r.get('gr1_realizable') is True)
    altl_real = sum(1 for r in results if r.get('atlas_ltl_realizable') is True)
    agr1_real = sum(1 for r in results if r.get('atlas_gr1_realizable') is True)

    print("Mined:        GR1Mine %d/%d    ATLAS[LTL] %d/%d    ATLAS[GR1] %d/%d" % (
        gr1_solved, total, altl_solved, total, agr1_solved, total))
    print("Realizable:   GR1Mine %d/%d    ATLAS[LTL] %d/%d    ATLAS[GR1] %d/%d" % (
        gr1_real, gr1_solved, altl_real, altl_solved, agr1_real, agr1_solved))

    # Compact table
    print()
    hdr = "%-42s | %4s | %8s %5s | %8s %5s | %8s %5s" % (
        'Benchmark', 'Vars',
        'GR1_t', 'Real',
        'A_LTL_t', 'Real',
        'A_GR1_t', 'Real')
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        def fmt_t(t):
            return ('%.1fs' % t) if t is not None else 'T/O'
        def fmt_r(v):
            if v is True: return 'Y'
            if v is False: return 'N'
            return '-'
        print("%-42s | %4d | %8s %5s | %8s %5s | %8s %5s" % (
            r['benchmark'][:42], r['num_vars'],
            fmt_t(r['gr1_time']), fmt_r(r.get('gr1_realizable')),
            fmt_t(r.get('atlas_ltl_time')), fmt_r(r.get('atlas_ltl_realizable')),
            fmt_t(r.get('atlas_gr1_time')), fmt_r(r.get('atlas_gr1_realizable'))))

    # Write CSV
    if results:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval_results.csv')
        if args.append_csv and os.path.exists(csv_path):
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writerows(results)
            print("\nResults appended to %s" % os.path.abspath(csv_path))
        else:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print("\nResults saved to %s" % os.path.abspath(csv_path))

    print("\nATLAS trace files in: %s" % tmpdir)


if __name__ == '__main__':
    main()
