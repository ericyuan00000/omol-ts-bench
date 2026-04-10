# OMol TS Bench

Benchmark suite for transition state (TS) optimization on organometallic reactions from the [OMol dataset](https://huggingface.co/datasets/facebook/OMol25), using [UMA](https://huggingface.co/facebook/UMA) as the force field.

## Pipeline

```
Reactant + Product
       │
       ▼
 NEB or GSM          ← double-ended TS search (guess)
       │
       ▼
  Sella TS opt       ← saddle-point refinement (order-1)
       │
       ▼
  Sella IRC          ← intrinsic reaction coordinate (both directions)
       │
       ▼
 OpenBabel validate  ← check IRC endpoints match reactant/product connectivity
```

Each stage produces a typed result dataclass (`NEBResult`, `SellaResult`, `IRCResult`, `ValidationResult`). The full pipeline result is a `PipelineResult`.

## Requirements

- Python ≥ 3.10
- CUDA GPU (UMA inference is GPU-only)
- [fairchem-core](https://github.com/facebookresearch/fairchem) ≥ 2.19
- [sella](https://github.com/zadorlab/sella) ≥ 2.4
- [ase](https://wiki.fysik.dtu.dk/ase/) ≥ 3.23
- openbabel-wheel ≥ 3.1

The UMA model weights are downloaded automatically from Hugging Face on first use and cached locally.

## Installation

```bash
conda create -n omol-ts-bench python=3.11
conda activate omol-ts-bench

# Core dependencies
pip install fairchem-core sella openbabel-wheel

# This package
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from ase.io import read
from omol_ts_bench.core.calculator import get_uma_calculator
from omol_ts_bench.core.pipeline import run_pipeline

calc = get_uma_calculator()          # uma-s-1p2, omol head
reactant = read("reactant.xyz")
product  = read("product.xyz")

# Set charge and spin (required by the OMol head)
for atoms in [reactant, product]:
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1

result = run_pipeline(
    reaction_id="rxn_0001",
    reactant=reactant,
    product=product,
    calc=calc,
)

print(result.error)                          # None on success
print(result.sella_result.ts_energy)        # eV
print(result.validation_result.both_match)  # True if IRC endpoints match
```

### Individual stages

```python
from omol_ts_bench.core.neb import run_neb
from omol_ts_bench.core.sella import run_sella
from omol_ts_bench.core.irc import run_irc
from omol_ts_bench.core.validate import validate_irc_endpoints

neb    = run_neb(reactant, product, calc, n_images=10, climb=True)
sella  = run_sella(neb.ts_guess, calc, fmax=0.02)
irc    = run_irc(sella.ts_optimized, calc, direction="both")
valid  = validate_irc_endpoints(reactant, product,
                                irc.endpoint_forward, irc.endpoint_reverse)
```

### CLI

```bash
# Run benchmark on all XYZ files in data/
omol-ts-bench run --data-dir data --method neb --output-dir results

# Filter by tier
omol-ts-bench run --data-dir data --tier smoke --registry configs/reactions.yaml

# Use multiple CPU workers (one calculator per worker process)
omol-ts-bench run --data-dir data --n-workers 4

# Print summary from a previous run
omol-ts-bench report results/results.json
```

## Data layout

```
data/
  xyz/
    reactants/
      rxn_0001.xyz
      rxn_0002.xyz
      ...
    products/
      rxn_0001.xyz    ← same stem as reactant
      rxn_0002.xyz
      ...
  references/         ← optional reference TS energies
```

Reactions are paired by filename stem. The loader skips any reaction whose product file is missing.

### Reaction registry

`configs/reactions.yaml` is an optional registry for tier assignment and metadata:

```yaml
reactions:
  - id: rxn_0001
    tier: smoke          # smoke | standard | full
    metal: Fe
    bond_type: C-H
    spin: 1
    n_atoms: 25
    difficulty: easy
```

Pass `--registry configs/reactions.yaml --tier smoke` to the CLI to filter by tier.

## Configuration

Default parameters for each stage live in `configs/default.yaml`. Pass a custom config with `--config`:

```yaml
neb:
  n_images: 10
  fmax: 0.05
  max_steps: 500
  climb: true
  interpolation: "idpp"   # linear | idpp
  optimizer: "BFGS"

sella:
  fmax: 0.02
  max_steps: 500
  order: 1
  delta: 0.01
  gamma: 0.4

irc:
  step_size: 0.1
  max_steps: 200
  fmax: 0.05
  direction: "both"       # both | forward | reverse
```

## Calculator

The default calculator is UMA-small v1.2 with the OMol head:

```python
from omol_ts_bench.core.calculator import get_uma_calculator

calc = get_uma_calculator(model_name="uma-s-1p2", task_name="omol")
```

Available models: `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`. Weights are fetched from `facebook/UMA` on Hugging Face and cached via `huggingface_hub`.

The OMol head reads `atoms.info["charge"]` and `atoms.info["spin"]` (spin multiplicity). Both must be set before calling `get_potential_energy()` or `get_forces()`.

## Workflow adapters

### Simple (multiprocessing)

Default. Runs reactions serially or across local CPU workers. Each worker reconstructs its own calculator from a dotted factory path so the model can be loaded per-process.

```bash
omol-ts-bench run --workflow simple --n-workers 4 ...
```

### QuAcc (HPC)

Distributes reactions across compute nodes via [QuAcc](https://quantum-accelerators.github.io/quacc/). Supports Parsl, Dask, Prefect, and other QuAcc backends.

```python
from omol_ts_bench.workflows.quacc import run_benchmark_quacc
from omol_ts_bench.core.calculator import get_uma_calculator

futures = run_benchmark_quacc(
    reactions=reactions,
    calc_factory=get_uma_calculator,
)
```

Configure the workflow engine before running:

```bash
quacc set WORKFLOW_ENGINE parsl
```

## Results

The simple workflow writes `results/results.json` — a list of per-reaction dicts:

```json
[
  {
    "reaction_id": "rxn_0001",
    "double_ended_method": "neb",
    "error": null,
    "neb_converged": true,
    "sella_converged": true,
    "irc_converged_forward": true,
    "irc_converged_reverse": true,
    "validation_both_match": true,
    "ts_energy": -42.31,
    "barrier_forward": 0.87
  }
]
```

Print a summary table:

```bash
omol-ts-bench report results/results.json
```

```
============================================================
OMol TS Benchmark Results Summary
============================================================
Total reactions:                    100
NEB converged:                      91 (91.0%)
Sella converged:                    85 (85.0%)
IRC both directions converged:      80 (80.0%)
Validated (endpoints match):        76 (76.0%)
Errors:                             9 (9.0%)
============================================================
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

Tests are split by concern and require no real reaction data:

| File | Requires UMA | Strategy |
|---|---|---|
| `test_calculator.py` | yes | Real UMA inference |
| `test_neb.py` | no | LennardJones + BFGS |
| `test_sella.py` | no | Mocked `sella.Sella` |
| `test_irc.py` | no | Mocked `sella.IRC` |
| `test_pipeline.py` | no | All stages mocked |
| `test_validate.py` | no | OpenBabel on H₂O |
| `test_interpolation.py` | no | Pure geometry |
| `test_gsm.py` | no | NotImplementedError check |

The UMA calculator fixture is session-scoped (`conftest.py`) so the model loads once per pytest run.

## Known limitations

- **OpenBabel connectivity validation** uses distance-based bond perception (`ConnectTheDots`). This works well for organic fragments but may misidentify bonds in metal coordination environments, haptic ligands, or highly strained geometries.
- **GSM** is not implemented. `run_de_gsm` raises `NotImplementedError`. A pyGSM–ASE bridge is planned.
- **Charge and spin must be set manually** on each `Atoms` object before running through the pipeline. Plain XYZ files loaded via `ase.io.read` will not have these fields.
