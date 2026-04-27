# GR(1) Specification Mining

A SAT-based tool for mining GR(1) (Generalized Reactivity of rank 1) specifications from lasso traces. Given positive and negative lasso-shaped traces, GR1Mine finds the smallest GR(1) formula that accepts all positive traces and rejects all negative traces.

The mined formula has the form:

```
(Init_e ∧ □Safety_e ∧ □◇J₁ ∧ ... ∧ □◇Jₘ) → (Init_s ∧ □Safety_s ∧ □◇G₁ ∧ ... ∧ □◇Gₖ)
```

where each component is a propositional formula over environment/system variables. The output is directly suitable for reactive synthesis (e.g., via [slugs](https://github.com/VerifiableRobotics/slugs)).

Built on top of [samples2LTL](https://github.com/ivan-gavran/samples2LTL) (Neider & Gavran, 2018).

## Setup

Requires Python 3.12 (3.14 breaks z3-solver build).

```bash
cd gr1-mining
python -m venv .venv
source .venv/bin/activate
pip install z3-solver==4.13.4.0 lark-parser pytictoc scikit-learn graphviz matplotlib
```

### External tools (for evaluation only)

**slugs** (GR(1) synthesis):
```bash
cd ..
git clone https://github.com/VerifiableRobotics/slugs.git
cd slugs/src && make
```

**ATLAS** (baseline LTL miner, Zhang et al. ICSE 2025): place the JAR at `../ATLAS/bin/Atlas.jar`. Requires Java.

**Spectra benchmarks**: 
```bash
cd ..
git clone https://github.com/SpectraSynthesizer/spectra-specs.git
```

## Usage

### Mining from a trace file

```bash
cd gr1-mining
source .venv/bin/activate
python experiments/gr1_experiment.py --traces <file> --max_depth 5
```

Options:
- `--max_depth D` — maximum number of literals per component (default: 5)
- `--num_justices M` — number of justice (assumption) conditions (default: 1)
- `--num_guarantees K` — number of guarantee conditions (default: 1)
- `--compare_baseline` — also run the baseline LTL miner for comparison

### Generating traces from a built-in spec

```bash
python experiments/gr1_experiment.py --generate --num_vars 3 --trace_length 5 --num_traces 20
```

### Trace file format

A trace file contains positive traces, then `---`, then negative traces, then `---`, then operators, then `---`, then max depth, then `---`, then the expected formula (optional).

Each trace is a semicolon-separated sequence of states, where each state is a comma-separated list of variable values (0 or 1). The `::` separator denotes the lasso start index (where the infinite loop begins).

Example trace with 3 variables and lasso start at position 2:
```
0,1,1;0,1,1;0,0,0;0,0,1;1,0,0::2
```

## Running Tests

```bash
cd gr1-mining
source .venv/bin/activate

# Core GR(1) encoding tests (12 tests)
python tests/test_gr1.py

# Spectra benchmark tests with primed variables (8 tests)
python tests/test_spectra.py
```

## Evaluation

### Main evaluation (Spectra benchmarks)

Runs GR1Mine against two ATLAS baselines on 60 Spectra benchmarks (AMBA bus arbiter and GenBuf protocol variants, 16-218 variables). Requires slugs, ATLAS JAR, and spectra-specs to be set up.

```bash
python experiments/eval_main.py
```

Options:
- `--miners gr1` — run only GR1Mine (skip ATLAS baselines)
- `--miners gr1,atlas_ltl` — run GR1Mine and ATLAS[LTL]
- `--miners atlas_gr1` — run only ATLAS[GR1]
- `--benchmarks amba_ahb` — filter benchmarks by name substring
- `--max-vars 50` — skip benchmarks with more than 50 variables
- `--append-csv` — append to existing `eval_results.csv` instead of overwriting

Results are saved to `eval_results.csv`.

### Synthetic evaluation

Runs GR1Mine vs samples2LTL[SAT] vs samples2LTL[DT] on 20 synthetic benchmarks with known GR(1) specs.

```bash
python experiments/evaluation.py
```

## How It Works

Unlike the baseline LTL miner which searches over all temporal operators (G, F, X, U), GR1Mine fixes the temporal skeleton to the GR(1) template and only searches for propositional content within each component. This eliminates the most expensive part of the SAT encoding (temporal operator semantics) and decomposes one large problem into several small independent propositional searches.

Key features:
- **Literal-leaves encoding**: each component is represented as a binary tree with D literal leaves (variable + polarity flag) and D-1 internal binary operator nodes (∧/∨). Negation is handled as a single boolean per leaf, not as a separate operator node.
- **Template enumeration**: the algorithm automatically discovers the right number of justice/guarantee conditions and whether init/safety components are needed.
- **Incremental solving**: structural constraints are encoded once per depth; only consistency constraints change across template configurations via push/pop.
- **Environment/system partition**: restricts which variables can appear in assumption vs. guarantee components, ensuring the mined spec is realizable.
