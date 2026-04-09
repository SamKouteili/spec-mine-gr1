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
- Synthetic evaluation: `python experiments/evaluation.py` (20 benchmarks, 5min timeout)
- Spectra evaluation: `python experiments/eval_spectra.py` (Spectra benchmarks, 5min timeout)
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
- Synthetic eval (20 benchmarks): GR1 20/20, SAT 17/20, DT results pending re-run with fix
- Spectra eval (12 benchmarks, 16-28 vars): GR1 12/12, SAT 12/12, DT 12/12
- DT is fastest on Spectra (~1s) but produces decision trees, not GR(1) specs
- SAT baseline often finds spurious separators (e.g., `x1 U ¬x1`) not usable for synthesis
- GR1Mine output is always synthesis-ready GR(1)
- Random trace generation produces few negative traces for large specs. Need biased generation.
- `evaluation_results.csv` and `spectra_eval_results.csv` in `gr1-mining/` have results
- Paper draft in `paper/` with FMCAD template, abstract, prelims, and evaluation sections

## Next Steps to Consider
- Integrate SAT-based trace generation into eval_spectra.py (moderate count ~30+20)
- CEGIS loop: mine candidate, verify against ground truth, add distinguishing traces, re-mine
- Biased negative trace generation (force assumptions to hold, then check guarantee failure)
- Run full 3-way evaluation (GR1 vs SAT vs DT) on synthetic benchmarks with DT fix applied
- Safety conjunct decomposition for large Spectra specs: mine G(φ₁) ∧ G(φ₂) as separate components
- Investigate why GR1Mine is slower than baselines on Spectra (template enumeration overhead at D=1)
- Paper: update evaluation section with corrected DT results, add Spectra benchmark table
