# GR(1) Specification Mining

## Project Structure
- `samples2LTL/` — Baseline Neider-Gavran 2018 LTL miner (untouched clone of github.com/ivan-gavran/samples2LTL)
- `gr1-mining/` — Our GR(1) extension (modified copy)
  - `smtEncoding/gr1SATEncoding.py` — Core GR(1) SAT encoding
  - `smtEncoding/dagSATEncoding.py` — Original baseline encoder (unchanged)
  - `utils/GR1Formula.py` — GR(1) formula representation + evaluation on lasso traces
  - `utils/spectra_parser.py` — Parser for Spectra (.spectra) benchmark files → GR1Formula
  - `experiments/gr1TestFileGeneration.py` — Random trace generator from known GR(1) specs
  - `experiments/gr1_experiment.py` — CLI entry point for GR(1) mining
  - `experiments/evaluation.py` — Synthetic benchmark suite (20 benchmarks, GR1 vs SAT vs DT)
  - `experiments/eval_spectra.py` — Spectra/Syntech benchmark suite (real GR(1) specs)
  - `experiments/plot_results.py` — Scatter/cactus plot generation for paper
  - `tests/test_gr1.py` — Test suite (12 tests: eval, mining, init, safety, multi-justice, template enumeration)
  - `tests/test_spectra.py` — Test suite (8 tests: primed vars, parser, Spectra file parsing, mining with next)
- `spectra-specs/` — Cloned Spectra benchmark repo (641 .spectra files)
- `paper/` — FMCAD paper draft

## How the Baseline Works (Neider-Gavran 2018)
Input: positive/negative lasso traces (ultimately periodic words). Output: smallest LTL formula separating them.
Encodes an unknown LTL formula as a syntax DAG of size D with SAT variables:
- `x[i][o]`: "node i is operator o" (searches over G, F, X, U, &, |, !, →, and prop vars)
- `l[i][j]`, `r[i][j]`: parent-child wiring
- `y[i][tr][t]`: "subformula i is true at position t in trace tr"
Constraints enforce valid DAG structure, LTL operator semantics at every trace position, and consistency (root true/false for positive/negative). Iterates D=1,2,3... until SAT.

## How Our GR(1) Miner Differs
We fix the temporal skeleton to the GR(1) template:
  `(Init_e ∧ □Safety_e(V,V') ∧ □◇J₁...□◇Jₘ) → (Init_s ∧ □Safety_s(V,V') ∧ □◇G₁...□◇Gₖ)`

Each component (init, safety, justice, guarantee) gets its own **independent mini-DAG** with only propositional operators (&, |, !). No temporal operator search at all.

The temporal semantics are hardcoded in the consistency constraints:
- Init: `holds[init][tr] ⟺ y[init][root][tr][0]` (check position 0)
- Safety: `holds[safety][tr] ⟺ ∧ y[safety][root][tr][t]` for all t (check all positions)
- Justice/Guarantee: `holds[J][tr] ⟺ ∨ y[J][root][tr][t]` for t in loop (some position in lasso loop)
- Positive traces: `assumptions → guarantees`
- Negative traces: `assumptions ∧ ¬guarantees`

This eliminates the G/F/X/U semantic constraints (the most expensive part of the baseline encoding) and decomposes one large SAT problem into several small independent propositional searches.

## Baselines
Three baselines available (all from samples2LTL):
- **samples2LTL[SAT]**: `DagSATEncoding` — SAT-based LTL mining over full operator set
- **samples2LTL[DT]**: `run_dt_solver` — Decision tree over SAT-mined LTL atoms (AtomBuilder + DTFormulaBuilder)
- Both invoked via `solverRuns.py`. DT requires `scikit-learn` and `graphviz` packages.

## External Benchmarks
- `spectra-specs/` — 641 Spectra specs from SpectraSynthesizer/spectra-specs GitHub
- `bloemDebugging/` (30 specs) and `cimattiAnalyzing/` (40 specs) are pure boolean, directly parseable
- All use `next()` in safety conditions — requires primed variable support
- Smallest specs: 16 vars (AMBA bus arbiter), largest attempted: 28 vars
- 24+ var specs often produce 0 negative traces from random sampling (vacuous satisfaction)

## Environment
- Python 3.12 (3.14 breaks z3-solver build)
- Venv at `gr1-mining/.venv/` — activate with `source gr1-mining/.venv/bin/activate`
- Key deps: `z3-solver==4.13.4.0`, `lark-parser`, `pytictoc`, `scikit-learn`, `graphviz`, `matplotlib`

## Running
- Tests: `cd gr1-mining && source .venv/bin/activate && python tests/test_gr1.py`
- Spectra tests: `python tests/test_spectra.py`
- **Main evaluation**: `python experiments/eval_main.py` (22 Spectra benchmarks, GR1Mine vs ATLAS[LTL] vs ATLAS[GR1], 5min timeout per miner)
- Synthetic evaluation: `python experiments/evaluation.py` (20 benchmarks, 5min timeout)
- Generate plots: `python experiments/plot_results.py`

## Encoder Interface
Any encoder must implement: `__init__(D, testTraces)`, `encodeFormula()`, `reconstructWholeFormula(model)`, `getInformativeVariables()`.
The solving loop in `formulaBuilder/satQuerying.py` and `solverRuns.py` is encoder-agnostic.

## Key Design Decisions
- Single shared D (DAG depth) across all components — avoids combinatorial explosion
- Template enumeration: `run_gr1_enumerating_solver()` in `gr1_experiment.py` iterates over (D, m, k, init, safety) configs — no template given as input
- Safety searches over current + primed (next-state) variables: `_get_variables_for_component()` returns 2n leaf vars for safety, n for others
- Primed variable labels use `xi'` convention (e.g., `x0'`, `x3'`), rendered as `next(xi)` in prettyPrint

## Critical Gotchas
- `Trace.intendedEvaluation` MUST be set (True for pos, False for neg) or DT's `buildAtoms` loops forever
- `multiprocessing.Queue.empty()` is unreliable in Python — never use it. Call `q.get()` directly or use return values
- When calling `run_dt_solver`, prefer direct call (q=None) over subprocess Queue pattern
- Random traces produce 0 genuine positives for Spectra benchmarks (16+ vars with complex safety) — all positives are vacuously true. Use SAT-based trace generation (`utils/trace_generator.py`) instead.
- Scaling trace count hurts GR1Mine more than baselines due to template enumeration overhead. DT baseline is essentially immune to trace count (~1s always).

## Current Status & Known Issues
- `utils/trace_generator.py` — SAT-based trace generation using Z3: constructs assumption-satisfying traces directly rather than random sampling. Produces genuine positives + assumption-satisfying negatives.
- `experiments/test_structured_traces.py` — Comparison script: random vs SAT-based traces across multiple trace counts/lengths
- GR1Formula has `evaluate_assumptions()` and `evaluate_guarantees()` methods for classifying traces by which side holds
- Full GR(1) template with next-state safety implemented and tested
- Template enumeration: miner discovers (m, k, init, safety) automatically
- GR1Mine output is always synthesis-ready GR(1)
- Paper draft in `paper/` with FMCAD template, abstract, prelims, algorithm section, and evaluation sections
- Old eval scripts (samples2LTL comparisons) moved to `gr1-mining/old_eval/`

### Slugs Integration Gotcha
- slugs writes "RESULT: Specification is realizable/unrealizable." to **stderr**, not stdout. Always check `result.stderr`. This caused a bug where all mined specs appeared unrealizable.

### Incremental Solving (branch: incremental-solving)
- Refactored `GR1SATEncoding` to support incremental solving: `encodeAllComponents()` encodes once, `_addConsistencyConstraints()` via push/pop per template config
- New `run_gr1_incremental_solver()` in `gr1_experiment.py` — 7-9× speedup over original enumeration on Spectra benchmarks (e.g., 22s → 2.9s on amba_ahb_fairness_1)
- Bottleneck was Python-side Z3 expression re-encoding, not SAT solving itself

### Symmetry Breaking (branch: symmetry-breaking)
- Commutative child ordering, double negation elimination, component permutation breaking
- No measurable speedup — confirmed the bottleneck was encoding, not SAT search

### Trace Generation: Key Finding
- A mix of SAT-generated traces + random traces (~25 each, ~50 total) is critical for a fair evaluation
- With only random traces, samples2LTL finds trivial spurious separators (e.g., `F x8`) at D=2 in <1s
- With mixed SAT+random traces, samples2LTL goes UNSAT from D=1 through D=6+ (each depth exponentially slower, D=5 takes ~55s, D=6 takes 800+s) while GR1Mine still solves at D=1 in <1s
- This is because SAT-generated traces are assumption-satisfying — they force any separator to capture the reactive assume-guarantee structure, which a flat LTL formula needs depth 7+ to express
- Removing U and -> from SAT baseline operators helps but doesn't change the fundamental scaling gap
- The prior eval used only random traces, which made both approaches look comparable — this was misleading

### Generalization Experiment
- `experiments/generalization_experiment.py` — compares GR1Mine vs SAT on held-out traces
- With 200 SAT-generated training traces, both achieve 89-99% accuracy on held-out — comparable
- The differentiator is not generalization accuracy but structural output (GR(1) vs flat LTL)

### Env/Sys Variable Partition (realizability constraint)
- Without partition, GR1Mine mines valid trace separators but often unrealizable specs — system variables leak into assumption justice conditions, making synthesized controllers impossible
- `GR1SATEncoding` now accepts optional `env_var_indices` parameter; when set, `_get_variables_for_component()` restricts leaf variables per component:
  - `safety_e` primed vars: env only | `safety_s` primed vars: sys only | current vars: all on both sides
  - Justice (J_i): env current vars only | Guarantees (G_i): sys current vars only
  - `init_e`: env vars only | `init_s`: sys vars only
- Threaded through `run_gr1_solver`, `run_gr1_enumerating_solver`, `run_gr1_incremental_solver`, and `eval_spectra.py`
- Benchmark (`experiments/bench_partition.py`): 3/4 unpartitioned runs had sys vars in justice assumptions; all partitioned runs clean. Partitioned is often faster (smaller search space) except when the constraint forces a higher depth
- Spectra parser already provides `metadata['env_vars']` / `metadata['sys_vars']` — no parser changes needed

## External Benchmarks: Spectra Coverage
- `spectra-specs/` has **662 .spectra files** across 10 subdirectories
- We use only **bloemDebugging/** (30 files) and **cimattiAnalyzing/** (40 files) — 22 benchmarks pass parsing + max_vars=30 filter
- These are all **boolean-only** specs with standard GR(1) syntax (AMBA bus arbiter + GenBuf protocol variants)
- **Not parseable** by our boolean-only parser:
  - SYNTECH15/17/19/20 (~552 files): enums, integers, defines, imports, PREV operator, typed variables (`Int(0..4)`, `{RED, GREEN, BLUE}`)
  - forklift (29 files): `respondsTo` pattern macros
  - CinderellaStepmother (9 files): `alw` keyword (alternative syntax for G)
  - amba/ (2 files): parametric specs with integer counters
- Extending the parser to handle enums/integers (by booleanizing them) would unlock SYNTECH families but is a significant effort

### Evaluation: ATLAS Baseline (current)
- **ATLAS** (Zhang et al., ICSE 2025): constrained LTL learner using Alloy/MaxSAT. Repo at `ATLAS/`, pre-built JAR at `ATLAS/bin/Atlas.jar`
- Two ATLAS configurations as baselines:
  - **ATLAS[LTL]**: unconstrained LTL mining (no template, no partition). Synthesis via ltlsynt.
  - **ATLAS[GR1]**: GR(1) template + env/sys partition constraint expressed in Alloy. Synthesis via slugs.
- ATLAS can express GR(1) + env/sys partition via Alloy constraints (`root in Imply`, `root.l in G`, subtree literal restrictions). However it still encodes full LTL temporal semantics internally, making it much slower.
- Preliminary results on `gen_buf_wo_ass_fairness_5_genbuf` (24 vars): GR1Mine 17.6s (REALIZABLE), ATLAS[LTL] 11.0s (REALIZABLE, flat LTL), ATLAS[GR1] TIMEOUT at 300s. GR1Mine massively outperforms ATLAS when GR(1) constraints are imposed.
- Main eval script: `experiments/eval_main.py` — runs all 22 Spectra benchmarks with GR1Mine, ATLAS[LTL], ATLAS[GR1], plus synthesis via slugs/ltlsynt.
- Trace file conversion: `write_atlas_unconstrained()` and `write_atlas_gr1()` in eval_main.py generate ATLAS-format .trace files from our Trace objects.
- ATLAS GR(1) output parsing: `atlas_gr1_to_slugs()` converts ATLAS prefix notation `"->(G(F(lhs)),G(F(rhs)))"` to slugs input format.
- **ltlsynt requires lowercase variable names** — spot treats uppercase as LTL operators. All var names lowercased in `synthesize_ltlsynt()`.
- Old evaluation scripts (samples2LTL[SAT], samples2LTL[DT] comparisons) archived in `gr1-mining/old_eval/`

## Next Steps to Consider
- **Run full evaluation** (`experiments/eval_main.py`) — GR1Mine vs ATLAS[LTL] vs ATLAS[GR1] on all 22 Spectra benchmarks. Expect ~5 hours.
- CEGIS loop: mine candidate, verify against ground truth, add distinguishing traces, re-mine
- Safety conjunct decomposition for large Spectra specs: mine G(φ₁) ∧ G(φ₂) as separate components
- Paper: write evaluation section with ATLAS comparison results
- Paper: add complexity analysis section comparing encoding size with baseline
- Investigate the 1 unrealizable GR1Mine result (specs_G5woef1): safety guarantee `G(!RtoB_ACK1)` constrains env var — tightening safety_s to sys-only current-state vars would fix but is more restrictive than standard GR(1)
