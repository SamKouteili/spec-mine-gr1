"""
Re-check realizability of previously mined GR1Mine specs.
Mines with same seeds as full_eval, but ONLY does synthesis check.
Skips SAT/DT baselines entirely.
"""
import sys, os, tempfile, subprocess, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.Traces import ExperimentTraces
from experiments.full_eval import generate_mixed_traces, mine_gr1, find_benchmarks
from experiments.synthesis_comparison import gr1_to_slugs

SLUGS_BIN = os.path.join(os.path.dirname(__file__), '..', '..', 'slugs', 'src', 'slugs')


def check_realizable(slugs_input):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slugsin', delete=False) as f:
        f.write(slugs_input)
        f.flush()
        tmp = f.name
    try:
        result = subprocess.run([SLUGS_BIN, tmp], capture_output=True, text=True, timeout=30)
        output = result.stderr  # slugs writes to stderr!
        if 'unrealizable' in output.lower():
            return False
        elif 'realizable' in output.lower():
            return True
        return None
    finally:
        os.unlink(tmp)


def main():
    # Read previous CSV to know which benchmarks had results
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'full_eval_results.csv')
    with open(csv_path) as f:
        prev_rows = {r['benchmark']: r for r in csv.DictReader(f)}

    benchmarks = {b['name']: b for b in find_benchmarks(max_vars=30)}

    print("%-45s | %4s | %8s | %10s | %s" % (
        'Benchmark', 'Vars', 'Time', 'Realizable', 'Formula'))
    print("-" * 120)

    real_count = 0
    unreal_count = 0
    solved_count = 0

    for name, prev in prev_rows.items():
        if prev['gr1_timeout'] == 'True' or prev.get('gr1_formula') in (None, 'None', ''):
            print("%-45s | %4s | %8s | %10s |" % (name, prev['num_vars'], 'T/O', '-'))
            continue

        bm = benchmarks.get(name)
        if not bm:
            continue

        spec = bm['spec']
        meta = bm['metadata']
        num_vars = meta['num_vars']
        env_indices = [meta['name_to_index'][v] for v in meta['env_vars']]
        nj, ng = len(spec.justices), len(spec.guarantees)

        # Re-mine with same seed (unavoidable since we need the formula object)
        train_pos, train_neg = generate_mixed_traces(
            spec, num_vars, 25, 25, seed=hash(name) % 10000)
        traces = ExperimentTraces(
            tracesToAccept=list(train_pos), tracesToReject=list(train_neg),
            operators=['&', '|', '!'])

        f, d, t, cfg = mine_gr1(
            traces, 5, min(nj + 1, 4), min(ng + 1, 4),
            env_indices, timeout=300)

        if f:
            solved_count += 1
            slugs_in = gr1_to_slugs(f, meta['env_vars'], meta['sys_vars'], meta['all_vars'])
            real = check_realizable(slugs_in)
            if real:
                real_count += 1
                status = 'REALIZABLE'
            elif real is False:
                unreal_count += 1
                status = 'UNREALIZABLE'
            else:
                status = 'ERROR'
            print("%-45s | %4d | %6.1fs | %10s | %s" % (
                name, num_vars, t, status, str(f)[:60]))
        else:
            print("%-45s | %4d | %8s | %10s |" % (name, num_vars, 'T/O', '-'))

        sys.stdout.flush()

    print()
    print("Solved: %d  |  Realizable: %d  |  Unrealizable: %d" % (
        solved_count, real_count, unreal_count))


if __name__ == '__main__':
    main()
