"""
CLI entry point for GR(1) specification mining.

Usage:
    python experiments/gr1_experiment.py --traces <file> --max_depth 5
    python experiments/gr1_experiment.py --traces <file> --max_depth 5 --num_justices 2
    python experiments/gr1_experiment.py --generate --num_vars 3 --trace_length 5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from utils.Traces import ExperimentTraces
from utils.SimpleTree import Formula
from utils.GR1Formula import GR1Formula
from smtEncoding.gr1SATEncoding import GR1SATEncoding
from experiments.gr1TestFileGeneration import generateTracesFromGR1


def generate_template_configs(max_justices=3, max_guarantees=3):
    """Generate template configurations in order of increasing complexity.

    Complexity is measured by total number of components:
      num_components = (1 if init else 0)*2 + (1 if safety else 0)*2 + m + k

    Within the same complexity, justice-only templates come first.
    """
    configs = []
    for include_init in [False, True]:
        for include_safety in [False, True]:
            for m in range(1, max_justices + 1):
                for k in range(1, max_guarantees + 1):
                    num_components = m + k
                    if include_init:
                        num_components += 2
                    if include_safety:
                        num_components += 2
                    configs.append((num_components, include_init, include_safety, m, k))
    configs.sort()
    return [(init, safety, m, k) for (_, init, safety, m, k) in configs]


def run_gr1_solver(traces, max_depth, num_justices=1, num_guarantees=1,
                   start_depth=1, step=1, include_init=False, include_safety=False,
                   env_var_indices=None):
    """Run the GR(1) miner with a fixed template, iterating over formula depths."""
    from pytictoc import TicToc
    t = TicToc()
    t.tic()

    for D in range(start_depth, max_depth + 1):
        logging.info("Trying D=%d..." % D)
        encoder = GR1SATEncoding(D, traces,
                                 num_justices=num_justices,
                                 num_guarantees=num_guarantees,
                                 include_init=include_init,
                                 include_safety=include_safety,
                                 env_var_indices=env_var_indices)
        encoder.encodeFormula()

        from z3 import sat
        if encoder.solver.check() == sat:
            model = encoder.solver.model()
            formula = encoder.reconstructWholeFormula(model)
            time_passed = t.tocvalue()
            return formula, D, time_passed

    time_passed = t.tocvalue()
    return None, max_depth, time_passed


def run_gr1_enumerating_solver(traces, max_depth, max_justices=3, max_guarantees=3,
                               env_var_indices=None):
    """Run the GR(1) miner, enumerating over all template configurations.

    For each propositional depth D = 1, 2, ..., tries all (m, k, init, safety)
    configurations in order of increasing complexity. Returns the first
    satisfying formula found.

    Also verifies soundness: the mined formula is evaluated on all traces.
    """
    from pytictoc import TicToc
    from z3 import sat
    t = TicToc()
    t.tic()

    configs = generate_template_configs(max_justices, max_guarantees)

    for D in range(1, max_depth + 1):
        for include_init, include_safety, m, k in configs:
            tmpl = ''
            if include_init: tmpl += 'I'
            if include_safety: tmpl += 'S'
            tmpl += 'J'
            logging.info("Trying D=%d, m=%d, k=%d, template=%s" % (D, m, k, tmpl))

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
                time_passed = t.tocvalue()

                # Soundness check
                sound = verify_formula(formula, traces)
                if not sound:
                    logging.warning("Soundness check FAILED for D=%d, m=%d, k=%d, template=%s" %
                                    (D, m, k, tmpl))
                    continue

                return formula, D, time_passed, (include_init, include_safety, m, k)

    time_passed = t.tocvalue()
    return None, max_depth, time_passed, None


def run_gr1_incremental_solver(traces, max_depth, max_justices=3, max_guarantees=3,
                               env_var_indices=None):
    """Run the GR(1) miner with incremental solving across template configs.

    For each depth D, encodes ALL possible components (max_justices justices,
    max_guarantees guarantees, init, safety) once. Then iterates over template
    configs using push/pop — only the consistency constraints change. This
    avoids re-encoding structural and semantic constraints ~40 times per D.

    Learned clauses from Z3's CDCL solver persist across push/pop, so
    information gained from one UNSAT config carries over to the next.
    """
    from pytictoc import TicToc
    from z3 import sat
    t = TicToc()
    t.tic()

    configs = generate_template_configs(max_justices, max_guarantees)

    # All possible component names
    all_justice_names = ['J_%d' % i for i in range(max_justices)]
    all_guarantee_names = ['G_%d' % i for i in range(max_guarantees)]

    for D in range(1, max_depth + 1):
        # Create one encoder with ALL possible components
        encoder = GR1SATEncoding(D, traces,
                                 num_justices=max_justices,
                                 num_guarantees=max_guarantees,
                                 include_init=True,
                                 include_safety=True,
                                 env_var_indices=env_var_indices)
        encoder.encodeAllComponents(unsatCore=True)
        encoder._addTemporalHoldsConstraints()

        for include_init, include_safety, m, k in configs:
            tmpl = ''
            if include_init: tmpl += 'I'
            if include_safety: tmpl += 'S'
            tmpl += 'J'
            logging.info("Trying D=%d, m=%d, k=%d, template=%s" % (D, m, k, tmpl))

            # Build active component lists for this config
            active_justices = all_justice_names[:m]
            active_guarantees = all_guarantee_names[:k]

            assumption_comps = active_justices[:]
            guarantee_comps = active_guarantees[:]
            if include_init:
                assumption_comps = ['init_e'] + assumption_comps
                guarantee_comps = ['init_s'] + guarantee_comps
            if include_safety:
                assumption_comps = ['safety_e'] + assumption_comps
                guarantee_comps = ['safety_s'] + guarantee_comps

            encoder.solver.push()
            encoder._addConsistencyConstraints(assumption_comps, guarantee_comps)

            if encoder.solver.check() == sat:
                model = encoder.solver.model()
                formula = encoder.reconstructForConfig(
                    model, active_justices, active_guarantees,
                    include_init, include_safety)

                encoder.solver.pop()

                sound = verify_formula(formula, traces)
                if not sound:
                    logging.warning("Soundness check FAILED for D=%d, m=%d, k=%d, template=%s" %
                                    (D, m, k, tmpl))
                    continue

                time_passed = t.tocvalue()
                return formula, D, time_passed, (include_init, include_safety, m, k)

            encoder.solver.pop()

    time_passed = t.tocvalue()
    return None, max_depth, time_passed, None


def verify_formula(formula, traces):
    """Verify that a mined formula correctly separates all traces."""
    for tr in traces.acceptedTraces:
        if not formula.evaluate_on_trace(tr):
            return False
    for tr in traces.rejectedTraces:
        if formula.evaluate_on_trace(tr):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='GR(1) Specification Mining')
    parser.add_argument('--traces', dest='tracesFileName', default=None,
                        help='Path to .trace file')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum propositional formula depth per component')
    parser.add_argument('--start_depth', type=int, default=1)
    parser.add_argument('--num_justices', type=int, default=1,
                        help='Number of justice (assumption) conditions')
    parser.add_argument('--num_guarantees', type=int, default=1,
                        help='Number of guarantee conditions')
    parser.add_argument('--generate', action='store_true',
                        help='Generate traces from a built-in GR(1) spec')
    parser.add_argument('--num_vars', type=int, default=2,
                        help='Number of variables (for --generate)')
    parser.add_argument('--trace_length', type=int, default=4,
                        help='Length of generated traces')
    parser.add_argument('--num_traces', type=int, default=20,
                        help='Number of traces to generate')
    parser.add_argument('--log', dest='loglevel', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel.upper())

    if args.generate:
        # Generate traces from a simple GR(1) spec: □◇x0 → □◇x1
        spec = GR1Formula(justices=[Formula('x0')], guarantees=[Formula('x1')])
        logging.info("Generating traces from: %s" % spec.prettyPrint())
        traces = generateTracesFromGR1(
            spec, args.trace_length,
            args.num_traces // 2, args.num_traces // 2,
            totalMax=args.num_traces * 10,
            numVars=args.num_vars
        )
        logging.info("Generated %d positive, %d negative traces" %
                     (len(traces.acceptedTraces), len(traces.rejectedTraces)))
    elif args.tracesFileName:
        traces = ExperimentTraces()
        traces.readTracesFromFile(args.tracesFileName)
    else:
        parser.error("Provide --traces <file> or --generate")

    # Run GR(1) miner
    logging.info("Running GR(1) miner (justices=%d, guarantees=%d)..." %
                 (args.num_justices, args.num_guarantees))
    formula, depth, time_gr1 = run_gr1_solver(
        traces, args.max_depth,
        num_justices=args.num_justices,
        num_guarantees=args.num_guarantees,
        start_depth=args.start_depth
    )

    if formula is not None:
        print("\n=== GR(1) Mining Result ===")
        print("Formula: %s" % formula.prettyPrint())
        print("Depth:   %d" % depth)
        print("Time:    %.3fs" % time_gr1)

        # Verify
        ok = True
        for tr in traces.acceptedTraces:
            if not formula.evaluate_on_trace(tr):
                ok = False
                break
        for tr in traces.rejectedTraces:
            if formula.evaluate_on_trace(tr):
                ok = False
                break
        print("Valid:   %s" % ok)
    else:
        print("\nGR(1) miner: UNSAT (no formula found up to depth %d)" % args.max_depth)
        print("Time: %.3fs" % time_gr1)



if __name__ == '__main__':
    main()
