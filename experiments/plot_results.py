"""Generate figures for the FMCAD paper from eval_results.csv."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TIMEOUT = 600

def load_results():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval_results.csv')
    results = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            results.append(r)
    return results


def scatter_plot(results, outdir):
    """Log-log scatter: GR1Mine vs ATLAS[LTL] and GR1Mine vs ATLAS[GR1]."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, key_t, key_to, label, color in [
        (ax1, 'atlas_ltl_time', 'atlas_ltl_timeout', 'ATLAS[LTL]', '#E67E22'),
        (ax2, 'atlas_gr1_time', 'atlas_gr1_timeout', 'ATLAS[GR1]', '#27AE60'),
    ]:
        gr1_solved, gr1_to, b_solved, b_to = [], [], [], []
        gr1_both, b_both = [], []

        for r in results:
            gr1_t = float(r['gr1_time']) if r['gr1_time'] else TIMEOUT
            b_t = float(r[key_t]) if r[key_t] else TIMEOUT
            gr1_timeout = r['gr1_timeout'] == 'True'
            b_timeout = r[key_to] == 'True'

            if not gr1_timeout and not b_timeout:
                gr1_both.append(gr1_t)
                b_both.append(b_t)
            elif not gr1_timeout and b_timeout:
                gr1_to.append(gr1_t)
                b_to.append(TIMEOUT)
            elif gr1_timeout and not b_timeout:
                gr1_solved.append(TIMEOUT)
                b_solved.append(b_t)

        gr1_both = np.array(gr1_both)
        b_both = np.array(b_both)

        ax.scatter(gr1_both, b_both,
                   c='#2C3E6B', s=45, zorder=5, label='Both solved')
        if gr1_to:
            ax.scatter(gr1_to, b_to,
                       c='#C0392B', s=45, marker='^', zorder=5,
                       label='%s timeout' % label)
        if gr1_solved:
            ax.scatter(gr1_solved, b_solved,
                       c='#8E44AD', s=45, marker='v', zorder=5,
                       label='GR1Mine timeout')

        lo, hi = 0.1, 1200
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
        ax.plot([lo, hi], [lo*10, hi*10], 'k:', lw=0.5, alpha=0.3)
        ax.plot([lo, hi], [lo*100, hi*100], 'k:', lw=0.5, alpha=0.3)

        ax.axhline(y=TIMEOUT, color='#C0392B', ls=':', lw=0.8, alpha=0.5)
        ax.axvline(x=TIMEOUT, color='#C0392B', ls=':', lw=0.8, alpha=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.3, 1200)
        ax.set_ylim(0.3, 1200)
        ax.set_xlabel('GR1Mine time (s)', fontsize=10)
        ax.set_ylabel('%s time (s)' % label, fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='lower right')
        ax.tick_params(labelsize=8)
        ax.set_title('GR1Mine vs %s' % label, fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'scatter.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'scatter.png'), bbox_inches='tight', dpi=200)
    print("Saved scatter plot")
    plt.close(fig)


def cactus_plot(results, outdir):
    """Cactus plot: cumulative solved vs time for all three tools."""
    fig, ax = plt.subplots(figsize=(5, 3.8))

    gr1_times = sorted([float(r['gr1_time']) for r in results
                        if r['gr1_time'] and r['gr1_timeout'] != 'True'])
    altl_times = sorted([float(r['atlas_ltl_time']) for r in results
                         if r['atlas_ltl_time'] and r['atlas_ltl_timeout'] != 'True'])
    agr1_times = sorted([float(r['atlas_gr1_time']) for r in results
                         if r['atlas_gr1_time'] and r['atlas_gr1_timeout'] != 'True'])

    n = len(results)

    for times, color, lbl, ls in [
        (gr1_times, '#2C3E6B', 'GR1Mine', '-'),
        (altl_times, '#E67E22', 'ATLAS[LTL]', '-'),
        (agr1_times, '#27AE60', 'ATLAS[GR1]', '-'),
    ]:
        if not times:
            continue
        ax.step(times, range(1, len(times) + 1),
                where='post', color=color, lw=2, ls=ls, label=lbl)
        # Extend flat line to timeout if not all solved
        if len(times) < n:
            ax.plot([times[-1], TIMEOUT],
                    [len(times), len(times)],
                    color=color, lw=2, ls='--', alpha=0.5)

    ax.axvline(x=TIMEOUT, color='#C0392B', ls=':', lw=0.8, alpha=0.5)
    ax.text(TIMEOUT * 0.65, 1, '10 min\ntimeout', fontsize=7,
            color='#C0392B', alpha=0.7, ha='right')

    ax.set_xscale('log')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Benchmarks solved', fontsize=10)
    ax.set_xlim(0.1, 900)
    ax.set_ylim(0, len(results) + 2)
    ax.set_yticks(range(0, len(results) + 2, 10))
    ax.legend(fontsize=9, loc='upper left')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'cactus.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'cactus.png'), bbox_inches='tight', dpi=200)
    print("Saved cactus plot")
    plt.close(fig)


def controller_states_plot(results, outdir):
    """Bar chart comparing controller sizes (states) for ATLAS[LTL] vs GR1Mine."""
    # Only show benchmarks where ATLAS[LTL] solved (13 cases)
    synth_csv = os.path.join(os.path.dirname(__file__), '..', 'synthesis_results.csv')
    if not os.path.exists(synth_csv):
        print("No synthesis_results.csv — skipping controller states plot")
        return

    synth = {}
    with open(synth_csv) as f:
        for r in csv.DictReader(f):
            synth[r['benchmark']] = r

    benchmarks, ltl_states, gr1_labels = [], [], []
    for r in results:
        b = r['benchmark']
        if b not in synth:
            continue
        s = synth[b]
        if s['altl_states'] and s['altl_states'] not in ('', '-', 'None'):
            ltl_st = int(s['altl_states'])
            gr1_st = s['gr1_states']
            if gr1_st in ('>300s', '', 'None', '-'):
                gr1_label = '>300s'
            else:
                gr1_label = str(int(gr1_st))

            short = b.replace('amba_ahb_', '').replace('specs_', '').replace('gen_buf_', 'gb_')
            short = short.replace('_genbuf', '').replace('_amba_ahb', '')
            benchmarks.append(short)
            ltl_states.append(ltl_st)
            gr1_labels.append(gr1_label)

    if not benchmarks:
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(benchmarks))
    width = 0.35

    ax.bar(x - width/2, ltl_states, width, color='#E67E22', label='ATLAS[LTL]')

    # For GR1Mine, show bars for known state counts, annotations for >300s
    gr1_vals = []
    for gl in gr1_labels:
        if gl == '>300s':
            gr1_vals.append(0)
        else:
            gr1_vals.append(int(gl))

    bars = ax.bar(x + width/2, gr1_vals, width, color='#2C3E6B', label='GR1Mine')
    for i, gl in enumerate(gr1_labels):
        if gl == '>300s':
            ax.annotate('>300s', (x[i] + width/2, 100),
                        fontsize=6, ha='center', rotation=90, color='#2C3E6B')

    ax.set_yscale('log')
    ax.set_ylabel('Controller states', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=6, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0.5, 200000)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'controller_states.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'controller_states.png'), bbox_inches='tight', dpi=200)
    print("Saved controller states plot")
    plt.close(fig)


if __name__ == '__main__':
    results = load_results()
    outdir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(outdir, exist_ok=True)
    scatter_plot(results, outdir)
    cactus_plot(results, outdir)
    controller_states_plot(results, outdir)
