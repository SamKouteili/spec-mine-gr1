"""
Re-run synthesis on all mined formulas from eval_results.csv.
Saves automaton outputs and collects state counts.
Does NOT re-mine — only synthesizes from already-mined formulas.
"""
import sys, os, csv, subprocess, tempfile, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SLUGS_BIN = os.path.join(os.path.dirname(__file__), '..', '..', 'slugs', 'src', 'slugs')
AUTOMATA_DIR = os.path.join(os.path.dirname(__file__), 'automata')
RESULTS_CSV = os.path.join(os.path.dirname(__file__), '..', 'eval_results.csv')
SYNTH_TIMEOUT = 60

# ============================================================
# ATLAS prefix → ltlsynt infix (for ATLAS[LTL])
# ============================================================

def _split_args(s):
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            return s[:i], s[i+1:]
    return s, ''


def atlas_prefix_to_ltlsynt(s, all_vars):
    s = s.strip()
    if s.startswith('->('):
        inner = s[3:-1]
        a, b = _split_args(inner)
        return '(%s -> %s)' % (atlas_prefix_to_ltlsynt(a, all_vars), atlas_prefix_to_ltlsynt(b, all_vars))
    elif s.startswith('G('):
        return 'G(%s)' % atlas_prefix_to_ltlsynt(s[2:-1], all_vars)
    elif s.startswith('F('):
        return 'F(%s)' % atlas_prefix_to_ltlsynt(s[2:-1], all_vars)
    elif s.startswith('X('):
        return 'X(%s)' % atlas_prefix_to_ltlsynt(s[2:-1], all_vars)
    elif s.startswith('U('):
        inner = s[2:-1]
        a, b = _split_args(inner)
        return '(%s U %s)' % (atlas_prefix_to_ltlsynt(a, all_vars), atlas_prefix_to_ltlsynt(b, all_vars))
    elif s.startswith('&('):
        inner = s[2:-1]
        a, b = _split_args(inner)
        return '(%s & %s)' % (atlas_prefix_to_ltlsynt(a, all_vars), atlas_prefix_to_ltlsynt(b, all_vars))
    elif s.startswith('|('):
        inner = s[2:-1]
        a, b = _split_args(inner)
        return '(%s | %s)' % (atlas_prefix_to_ltlsynt(a, all_vars), atlas_prefix_to_ltlsynt(b, all_vars))
    elif s.startswith('!('):
        return '!(%s)' % atlas_prefix_to_ltlsynt(s[2:-1], all_vars)
    elif s.startswith('x'):
        idx = int(s[1:])
        return all_vars[idx].lower() if idx < len(all_vars) else s
    return s


def run_ltlsynt(formula_str, env_vars, sys_vars, all_vars, outpath):
    ltl_str = atlas_prefix_to_ltlsynt(formula_str, all_vars)
    env_lc = [v.lower() for v in env_vars]
    sys_lc = [v.lower() for v in sys_vars]
    try:
        result = subprocess.run(
            ['ltlsynt',
             '--ins=' + ','.join(env_lc),
             '--outs=' + ','.join(sys_lc),
             '-f', ltl_str],
            capture_output=True, text=True, timeout=SYNTH_TIMEOUT)
        output = result.stdout
        with open(outpath, 'w') as f:
            f.write(output)
        # Extract state count
        m = re.search(r'States:\s*(\d+)', output)
        states = int(m.group(1)) if m else None
        if 'UNREALIZABLE' in output:
            return 'UNREALIZABLE', states
        elif 'REALIZABLE' in output:
            return 'REALIZABLE', states
        return 'UNKNOWN', states
    except subprocess.TimeoutExpired:
        with open(outpath, 'w') as f:
            f.write('TIMEOUT\n')
        return 'TIMEOUT', None


# ============================================================
# ATLAS[GR1] prefix → slugs (for ATLAS[GR1])
# ============================================================

def atlas_prefix_to_slugs_prop(s, all_vars):
    """Convert ATLAS prefix prop formula to slugs prefix notation."""
    s = s.strip()
    if s.startswith('&('):
        inner = s[2:-1]
        a, b = _split_args(inner)
        return '& %s %s' % (atlas_prefix_to_slugs_prop(a, all_vars), atlas_prefix_to_slugs_prop(b, all_vars))
    elif s.startswith('|('):
        inner = s[2:-1]
        a, b = _split_args(inner)
        return '| %s %s' % (atlas_prefix_to_slugs_prop(a, all_vars), atlas_prefix_to_slugs_prop(b, all_vars))
    elif s.startswith('!('):
        return '! %s' % atlas_prefix_to_slugs_prop(s[2:-1], all_vars)
    elif s.startswith('x'):
        idx = int(s[1:])
        return all_vars[idx] if idx < len(all_vars) else s
    return s


def atlas_gr1_to_slugs_input(formula_str, env_vars, sys_vars):
    all_vars = env_vars + sys_vars
    lines = ['[INPUT]'] + env_vars + ['', '[OUTPUT]'] + sys_vars + ['']
    lines += ['[ENV_INIT]', '', '[ENV_TRANS]', '']

    if formula_str.startswith('->(G(F('):
        inner = formula_str[3:-1]  # G(F(lhs)),G(F(rhs))
        lhs_gf, rhs_gf = _split_args(inner)
        lhs = lhs_gf[4:-2]  # strip G(F( and ))
        rhs = rhs_gf[4:-2]
        lines += ['[ENV_LIVENESS]', atlas_prefix_to_slugs_prop(lhs, all_vars), '']
        lines += ['[SYS_INIT]', '', '[SYS_TRANS]', '']
        lines += ['[SYS_LIVENESS]', atlas_prefix_to_slugs_prop(rhs, all_vars), '']
    else:
        return None
    return '\n'.join(lines)


# ============================================================
# GR1Mine formula string → slugs
# ============================================================

def parse_gr1mine_to_slugs(formula_str, env_vars, sys_vars):
    """Parse GR1Mine display string into slugs input format."""
    all_vars = env_vars + sys_vars

    # Split on → to get assumptions and guarantees
    parts = formula_str.split(' → ')
    if len(parts) == 2:
        ass_str, guar_str = parts
    elif len(parts) == 1:
        # No assumptions (just guarantees)
        ass_str, guar_str = '', parts[0]
    else:
        # Multiple →, rejoin
        ass_str = parts[0]
        guar_str = ' → '.join(parts[1:])

    def split_top_level_conj(s):
        """Split on top-level ∧ (conjunction at the GR1 level)."""
        s = s.strip()
        # Remove outer parens if they wrap everything
        if s.startswith('(') and s.endswith(')'):
            # Check if these parens wrap the whole thing
            depth = 0
            wraps_all = True
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(s) - 1:
                    wraps_all = False
                    break
            if wraps_all:
                s = s[1:-1]

        components = []
        depth = 0
        bracket_depth = 0
        current = ''
        i = 0
        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                depth -= 1
                current += c
            elif c == '[':
                bracket_depth += 1
                current += c
            elif c == ']':
                bracket_depth -= 1
                current += c
            elif s[i:i+3] == ' ∧ ' and depth == 0 and bracket_depth == 0:
                components.append(current.strip())
                current = ''
                i += 3
                continue
            else:
                current += c
            i += 1
        if current.strip():
            components.append(current.strip())
        return components

    def classify_component(s):
        """Classify and extract propositional content."""
        s = s.strip()
        if s.startswith('[]<>'):
            # Justice or guarantee: []<>(prop)
            inner = s[4:]
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
            return 'liveness', inner
        elif s.startswith('[]'):
            # Safety: [](prop)
            inner = s[2:]
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
            return 'safety', inner
        else:
            # Init: bare propositional
            return 'init', s

    def prop_to_slugs_prefix(s):
        """Convert GR1Mine infix propositional formula to slugs prefix."""
        s = s.strip()
        # Remove outer parens
        while s.startswith('(') and s.endswith(')'):
            depth = 0
            wraps_all = True
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(s) - 1:
                    wraps_all = False
                    break
            if wraps_all:
                s = s[1:-1].strip()
            else:
                break

        # Handle next(xi) → xi'
        m = re.match(r'^next\(x(\d+)\)$', s)
        if m:
            idx = int(m.group(1))
            return (all_vars[idx] if idx < len(all_vars) else 'x%d' % idx) + "'"

        # Handle xi
        m = re.match(r'^x(\d+)$', s)
        if m:
            idx = int(m.group(1))
            return all_vars[idx] if idx < len(all_vars) else s

        # Handle negation: ! expr
        if s.startswith('! '):
            return '! %s' % prop_to_slugs_prefix(s[2:])

        # Handle (! expr) — but verify the outer parens actually match
        if s.startswith('(! ') and s.endswith(')'):
            # Check that the ( at position 0 matches the ) at the end
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0:
                    if i == len(s) - 1:
                        return '! %s' % prop_to_slugs_prefix(s[3:-1])
                    break

        # Find top-level binary operator: | or &
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0:
                if s[i:i+3] == ' | ':
                    left = s[:i].strip()
                    right = s[i+3:].strip()
                    return '| %s %s' % (prop_to_slugs_prefix(left), prop_to_slugs_prefix(right))
                elif s[i:i+3] == ' & ':
                    left = s[:i].strip()
                    right = s[i+3:].strip()
                    return '& %s %s' % (prop_to_slugs_prefix(left), prop_to_slugs_prefix(right))

        return s  # fallback

    ass_components = split_top_level_conj(ass_str) if ass_str.strip() else []
    guar_components = split_top_level_conj(guar_str) if guar_str.strip() else []

    env_init, env_safety, env_liveness = [], [], []
    sys_init, sys_safety, sys_liveness = [], [], []

    for comp in ass_components:
        kind, prop = classify_component(comp)
        prefix = prop_to_slugs_prefix(prop)
        if kind == 'init':
            env_init.append(prefix)
        elif kind == 'safety':
            env_safety.append(prefix)
        elif kind == 'liveness':
            env_liveness.append(prefix)

    for comp in guar_components:
        kind, prop = classify_component(comp)
        prefix = prop_to_slugs_prefix(prop)
        if kind == 'init':
            sys_init.append(prefix)
        elif kind == 'safety':
            sys_safety.append(prefix)
        elif kind == 'liveness':
            sys_liveness.append(prefix)

    lines = ['[INPUT]'] + env_vars + ['', '[OUTPUT]'] + sys_vars + ['']
    lines += ['[ENV_INIT]'] + env_init + ['']
    lines += ['[ENV_TRANS]'] + env_safety + ['']
    lines += ['[ENV_LIVENESS]'] + env_liveness + ['']
    lines += ['[SYS_INIT]'] + sys_init + ['']
    lines += ['[SYS_TRANS]'] + sys_safety + ['']
    lines += ['[SYS_LIVENESS]'] + sys_liveness + ['']
    return '\n'.join(lines)


def run_slugs(slugs_input, outpath):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slugsin', delete=False) as f:
        f.write(slugs_input)
        f.flush()
        tmp = f.name
    try:
        # Check realizability + extract symbolic strategy (BDD)
        bdd_path = outpath.replace('.txt', '.bdd')
        result = subprocess.run(
            [SLUGS_BIN, tmp, '--symbolicStrategy', bdd_path],
            capture_output=True, text=True, timeout=SYNTH_TIMEOUT)
        stderr = result.stderr.lower()

        if 'unrealizable' in stderr:
            with open(outpath, 'w') as f:
                f.write('UNREALIZABLE\n' + result.stderr)
            return 'UNREALIZABLE', None

        if 'realizable' not in stderr:
            with open(outpath, 'w') as f:
                f.write('UNKNOWN\n' + result.stderr)
            return 'UNKNOWN', None

        # Extract BDD node count
        bdd_nodes = None
        if os.path.exists(bdd_path) and os.path.getsize(bdd_path) > 0:
            with open(bdd_path) as bf:
                m = re.search(r'\.nnodes\s+(\d+)', bf.read())
                if m:
                    bdd_nodes = int(m.group(1))

        with open(outpath, 'w') as f:
            f.write('REALIZABLE\n')
            f.write('BDD nodes: %s\n' % bdd_nodes)
            f.write(result.stderr)

        return 'REALIZABLE', bdd_nodes

    except subprocess.TimeoutExpired:
        with open(outpath, 'w') as f:
            f.write('TIMEOUT\n')
        return 'TIMEOUT', None
    finally:
        os.unlink(tmp)


# ============================================================
# Main
# ============================================================

def main():
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        bench = row['benchmark']
        nv = int(row['num_vars'])
        ne = int(row['num_env'])
        env_vars = ['x%d' % i for i in range(ne)]
        sys_vars = ['x%d' % i for i in range(ne, nv)]
        all_vars = env_vars + sys_vars

        gr1_formula = row['gr1_formula']
        atlas_ltl_formula = row['atlas_ltl_formula']
        atlas_gr1_formula = row['atlas_gr1_formula']

        entry = {'benchmark': bench, 'num_vars': nv}
        print('=== %s (vars=%d) ===' % (bench, nv))
        sys.stdout.flush()

        # --- GR1Mine → slugs ---
        if gr1_formula:
            outpath = os.path.join(AUTOMATA_DIR, 'gr1mine', bench + '.txt')
            slugs_path = os.path.join(AUTOMATA_DIR, 'gr1mine', bench + '.slugsin')
            slugs_input = parse_gr1mine_to_slugs(gr1_formula, env_vars, sys_vars)
            if slugs_input:
                with open(slugs_path, 'w') as f:
                    f.write(slugs_input)
                real, states = run_slugs(slugs_input, outpath)
                entry['gr1_real'] = real
                entry['gr1_states'] = states
                entry['gr1_path'] = os.path.relpath(outpath, os.path.dirname(RESULTS_CSV))
                print('  GR1Mine: %s  states=%s' % (real, states))
            else:
                entry['gr1_real'] = 'PARSE_ERR'
                entry['gr1_states'] = None
                entry['gr1_path'] = ''
                print('  GR1Mine: PARSE ERROR')
        else:
            entry['gr1_real'] = ''
            entry['gr1_states'] = None
            entry['gr1_path'] = ''

        # --- ATLAS[LTL] → ltlsynt ---
        if atlas_ltl_formula:
            outpath = os.path.join(AUTOMATA_DIR, 'atlas_ltl', bench + '.hoa')
            real, states = run_ltlsynt(atlas_ltl_formula, env_vars, sys_vars, all_vars, outpath)
            entry['altl_real'] = real
            entry['altl_states'] = states
            entry['altl_path'] = os.path.relpath(outpath, os.path.dirname(RESULTS_CSV))
            print('  ATLAS[LTL]: %s  states=%s' % (real, states))
        else:
            entry['altl_real'] = ''
            entry['altl_states'] = None
            entry['altl_path'] = ''

        # --- ATLAS[GR1] → slugs ---
        if atlas_gr1_formula:
            outpath = os.path.join(AUTOMATA_DIR, 'atlas_gr1', bench + '.txt')
            slugs_path = os.path.join(AUTOMATA_DIR, 'atlas_gr1', bench + '.slugsin')
            slugs_input = atlas_gr1_to_slugs_input(atlas_gr1_formula, env_vars, sys_vars)
            if slugs_input:
                with open(slugs_path, 'w') as f:
                    f.write(slugs_input)
                real, states = run_slugs(slugs_input, outpath)
                entry['agr1_real'] = real
                entry['agr1_states'] = states
                entry['agr1_path'] = os.path.relpath(outpath, os.path.dirname(RESULTS_CSV))
                print('  ATLAS[GR1]: %s  states=%s' % (real, states))
            else:
                entry['agr1_real'] = 'PARSE_ERR'
                entry['agr1_states'] = None
                entry['agr1_path'] = ''
                print('  ATLAS[GR1]: PARSE ERROR')
        else:
            entry['agr1_real'] = ''
            entry['agr1_states'] = None
            entry['agr1_path'] = ''

        results.append(entry)
        print()
        sys.stdout.flush()

    # Print summary table
    print('=' * 130)
    print('%-45s | %4s | %12s %8s | %12s %8s | %12s %8s' % (
        'Benchmark', 'Vars',
        'GR1Mine', 'BDD',
        'ATLAS[LTL]', 'States',
        'ATLAS[GR1]', 'BDD'))
    print('-' * 130)
    for e in results:
        gr1_s = '%s' % (e['gr1_states'] if e['gr1_states'] is not None else '-')
        altl_s = '%s' % (e['altl_states'] if e['altl_states'] is not None else '-')
        agr1_s = '%s' % (e['agr1_states'] if e['agr1_states'] is not None else '-')
        print('%-45s | %4d | %12s %8s | %12s %8s | %12s %8s' % (
            e['benchmark'], e['num_vars'],
            e.get('gr1_real', '-'), gr1_s,
            e.get('altl_real', '-'), altl_s,
            e.get('agr1_real', '-'), agr1_s))

    # Save to CSV
    out_csv = os.path.join(os.path.dirname(RESULTS_CSV), 'synthesis_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'benchmark', 'num_vars',
            'gr1_real', 'gr1_states', 'gr1_path',
            'altl_real', 'altl_states', 'altl_path',
            'agr1_real', 'agr1_states', 'agr1_path'])
        w.writeheader()
        w.writerows(results)
    print('\nSaved to', out_csv)


if __name__ == '__main__':
    main()
