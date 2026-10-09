# DEComp Benchmark

DEComp is a benchmark for comparing equation discovery frameworks on the
same synthetic ODE/PDE datasets. The current working comparison focuses on
EPDE, VWSR, PySINDy, DeepMoD, DISCOVER and EDL.

## What Is Compared

The benchmark uses the datasets and true coefficients from
`data/config.py`. Each framework is evaluated on the same loaded data from
`utils/dataloader.py`.

DEComp provides two complementary comparison protocols:

- **`fixed`** (default) supplies the same NumPy derivatives and the same finite
  candidate library to every compatible optimizer. It isolates sparse term
  selection and coefficient estimation.
- **`native`** keeps the datasets and metrics common, but lets each complete
  framework use its own preprocessing, differentiation, library representation,
  and search procedure. Broad complexity limits are aligned; any small
  dataset-level stability overrides are explicit in `data/config.py`.

The native adapter contract supplies raw observations, coordinates, dependent
variable names, the requested target derivative, and a broad maximum derivative
order. It does not supply the fixed feature matrix, fixed derivatives, or
hidden optimizer settings. Framework limitations are reported as `unsupported`
instead of being emulated with the fixed pipeline.

The protocols answer different questions: `fixed` compares optimizers under
controlled inputs, while `native` compares practical framework pipelines.

| Framework | `native` coverage |
|---|---|
| PySINDy | all configured datasets |
| DeepMoD | scalar 1D PDE, first-order time target |
| EPDE | all configured datasets |
| DISCOVER | scalar ODE and scalar 1D PDE |
| EDL | scalar ODE and scalar 1D PDE; external LLM opt-in |
| VWSR | not applicable: sparse estimator only |

## Framework Integration

The benchmark always keeps data loading and metric calculation common. In the
`fixed` protocol it also centralizes derivatives and library construction; in
`native`, those stages belong to each framework's own pipeline.

- **PySINDy** is used as the direct sparse-regression baseline. In `fixed`, the wrapper
  builds the shared candidate library from `utils/sindy_library.py`, computes
  derivatives with `utils/derivatives.py`, and applies the optimizer configured
  in `data/config.py`. In `native`, derivatives come from PySINDy's
  `FiniteDifference`. ODE and system spaces are assembled entirely from
  PySINDy's `PolynomialLibrary`, `CustomLibrary`, and `GeneralizedLibrary`.
  Scalar PDEs combine `PDELibrary` with the benchmark-only coordinate and
  compound terms through `GeneralizedLibrary`; Navier-Stokes keeps its compact
  specialized terms. Candidate counts match or exceed `fixed` without reusing
  its precomputed feature matrix. STLSQ is the default sparse optimizer; the few
  clean-stability overrides that use FROLS or different STLSQ scaling are listed
  in `NATIVE_PYSINDY_DATASET_OVERRIDES`.
- **DeepMoD** is run through its sparse estimators on the same precomputed
  NumPy feature matrices and target derivatives. The wrapper uses DeePyMoD's
  optimizer classes, then maps selected terms back to the shared benchmark
  feature names for metrics. Its `native` path trains the original neural
  approximation, differentiates it with autograd, builds `Library1D`, and uses
  DeePyMoD's constraint and sparsity scheduler. The stock path currently covers
  scalar 1D PDEs whose target is the first time derivative.
- **EPDE** is run through its evolutionary search pipeline. Its source is
  checked out as the `epde/EPDE` submodule. Dataset-specific EPDE settings are
  stored directly in `data/config.py`; the wrapper prepares benchmark data,
  coordinate tensors, derivatives, and custom token families so EPDE searches
  in the comparable term space. In `native`, EPDE receives raw data, computes
  derivatives with its own preprocessor, and searches standard EPDE tokens.
- **VWSR** is evaluated separately from EPDE's evolutionary search. The
  `vwsr/run.py` wrapper imports only EPDE's variance-weighted sparse-regression
  estimator and applies it directly to the shared fixed feature matrices and
  precomputed derivatives. Its per-term penalties are derived from spatial or
  temporal variation of locally varying coefficient estimates. VWSR has no
  `native` protocol because it is an estimator, not a complete data-to-equation
  framework; this is reported as `unsupported`.
- **DISCOVER** is connected as a separate submodule and container because it
  uses a TensorFlow 1.x stack. The benchmark uses its external fixed-library
  mode: shared features are passed as fixed `theta_*` tokens, and DISCOVER
  searches symbolic combinations of those tokens. In `native`, DISCOVER uses
  its own finite differences and standard symbolic grammar on raw scalar ODE or
  1D PDE data. Systems are not supported.
- **EDL** is connected as the `edl/EDL` submodule. The original EDL method uses
  an LLM to propose equation candidates and then scores/fits them on data. For
  reproducible `fixed` runs without API keys, `edl/run.py` uses EDL's STRidge
  sparse-regression backend on the same fixed feature matrices and target
  derivatives built by the shared `utils/` layer. The `native` wrapper prepares
  EDL's original LLM loop, but it reports `skipped` unless external calls are
  explicitly enabled and an API key is present. The adapter serializes EDL's
  module-level evaluator hook, so native EDL searches within one process run
  sequentially.

## Main Scripts

`clean_run_metrics.py` measures clean-data runs for one framework:

```powershell
python clean_run_metrics.py pysindy
python clean_run_metrics.py pysindy --protocol native --datasets burgers_data.mat
python clean_run_metrics.py deepmod --protocol native --datasets burgers_data.mat
```

The output CSV contains runtime, library size, discovered active terms, expected
terms, and coefficient diagnostics. `coefficient_error` is reported only when
the discovered structure is correct; `relative_error_sum` is the diagnostic
error for the best structural match and can also be present for an incorrect
structure.

`noise_test.py` is the runner with noisy data:

```powershell
python noise_test.py pysindy --datasets ode_data.npy --levels 0.5 0.75 1.0
python noise_test.py deepmod --datasets ac_data.npy --levels 10 15 20
python noise_test.py vwsr --datasets kdv_data.mat --levels 0.001 0.005 0.01
python noise_test.py discover --protocol native --datasets kdv_data.mat --levels 0.01 0.05 --runs 5
```

Noise is Gaussian and proportional to the data standard deviation:

```text
u_noisy = u + noise_level * 0.01 * std(u) * np.random.normal()
```

By default, a run is successful only when it matches an accepted ground-truth
structure. `--success-reference clean` retains the older repeatability test.
For systems, `__system__` is correct only when every component is correct.

`noise_boundary_metrics.py` measures structural and coefficient errors at fixed
noise boundaries:

```powershell
python noise_boundary_metrics.py pysindy --boundaries-csv results\pysindy_noisy\noise_manual_3_5_summary.csv
python noise_boundary_metrics.py deepmod --boundaries-csv results\deepmod\noise_manual_3_5_summary_noise_tuned.csv
python noise_boundary_metrics.py vwsr
```

It reports HD across all noisy runs and RE only for structurally correct runs.
For native results, pass a boundary file obtained from a native sweep and add
`--protocol native`.

`run_benchmark.py` runs a protocol across the isolated framework containers:

```powershell
python run_benchmark.py clean --protocol native --frameworks pysindy deepmod epde discover
python run_benchmark.py noise --protocol native --frameworks pysindy deepmod epde discover --datasets burgers_data.mat --levels 0 0.1 0.5 --runs 5
```

The unrestricted all-framework native command returns non-zero for EDL without
API opt-in and for VWSR, which has no complete native pipeline. Pass
`--allow-empty` only when those capability statuses are expected.

EDL's external LLM is not contacted unless `--allow-external-llm` is present.
The EDL container then reads `EDL_API_KEY` or `OPENAI_API_KEY` from the
environment.

Noise generation and framework stochasticity use separate seeds. Noise
realizations vary with the run index, while `--algorithm-seed` (default `0`) is
held fixed so optimizer randomness is not mixed into the noise axis.

EPDE normally returns the configured `--solution-index`, selected without
ground-truth access. `--epde-pareto-oracle` is an explicitly diagnostic upper
bound that chooses among Pareto candidates using the known equation; its CSV
rows are marked `selection_policy=ground_truth_pareto_oracle` and must not be
used as primary framework results.

`--native-max-iterations` is useful for smoke tests. It limits neural-network
iterations in DeepMoD, evolutionary epochs in EPDE, generated samples in
DISCOVER, and LLM refinement epochs in EDL. It is a runtime cap, not a shared
algorithmic hyperparameter. DISCOVER requires at least one complete policy
batch, so its native value must be at least `250` with the default settings.

`deepmod/run.py`, `epde/run.py`, `vwsr/run.py`, `discover/run.py`, and
`edl/run.py` are benchmark wrappers.
Their framework sources are checked out as git submodules in
`deepmod/deepymod/`, `epde/EPDE/`, `discover/discover/`, and `edl/EDL/`.
DISCOVER supports the configured scalar ODE and 1D PDE datasets through an
external fixed-library mode. In this mode the shared
`utils/` layer loads the data, computes the target derivative, and builds the
same candidate library used by the other frameworks. DISCOVER then searches over
fixed `theta_*` library terms instead of relying on its built-in PDE derivative
tokens. For ODE datasets, the wrapper passes a dummy spatial coordinate only to
fit DISCOVER's PDE-task interface; the actual ODE library and target are still
computed from the benchmark data.

In `fixed`, EDL supports the configured benchmark datasets through its STRidge
backend. Native EDL metrics use the LLM wrapper described above and remain
disabled unless explicitly requested because they require credentials and incur
external, non-deterministic calls.

## Docker

Each framework has its own Docker image and Python environment:

```text
pysindy   -> PySINDy dependencies
deepmod   -> DeePyMoD dependencies
epde      -> EPDE dependencies
vwsr      -> EPDE's standalone VWSR sparse optimizer
discover  -> DISCOVER with TensorFlow 1.x
edl       -> EDL sparse-regression backend
```

Build all images:

```powershell
docker compose build
```

Run the default clean metrics for one framework:

```powershell
docker compose run --rm pysindy
docker compose run --rm deepmod
docker compose run --rm epde
docker compose run --rm vwsr
docker compose run --rm discover
docker compose run --rm edl
```

Any benchmark script can be run in the matching framework container:

```powershell
docker compose run --rm pysindy python noise_test.py pysindy --datasets ode_data.npy --levels 0.5 1.0
docker compose run --rm deepmod python noise_test.py deepmod --datasets ac_data.npy --levels 10 15 20
docker compose run --rm epde python clean_run_metrics.py epde --datasets wave_data.csv
docker compose run --rm vwsr python clean_run_metrics.py vwsr --datasets kdv_data.mat
docker compose run --rm discover python noise_boundary_metrics.py discover --boundaries-csv results/discover/noise_boundaries_3_5.csv
docker compose run --rm edl python clean_run_metrics.py edl --datasets burgers_data.mat
docker compose run --rm deepmod python clean_run_metrics.py deepmod --protocol native --datasets burgers_data.mat
```

All framework containers mount `data/` as read-only and write outputs to the
shared `results/` directory. DISCOVER stays separate because it depends on the
old TensorFlow 1.x stack, which conflicts with the modern environments used by
the other frameworks.

The supplied DeepMoD and EPDE images use the official CPU build of PyTorch.
GPU runs require a CUDA-enabled image and a matching Compose device override.

## Metrics

Clean runs report:

- `runtime_seconds`: wall-clock time for one clean run;
- `library_size`: number of candidate terms;
- `relative_error_sum`: diagnostic sum of relative coefficient errors for the
  best ground-truth match;
- `coefficient_error`: the same coefficient metric, reported only for a
  structurally correct equation;
- `missing_terms` and `extra_terms`: structural differences against ground truth.

Every CSV also includes `protocol` and `status`. Status is `ok`, `unsupported`,
`skipped`, or `error`, so an unavailable native capability or disabled external
service cannot be mistaken for failed equation recovery.
Metric commands return a non-zero exit code when at least one row has
`status=error`. They also return code `2` when no row has `status=ok`, preventing
an entirely skipped or unsupported invocation from being reported as a
successful measurement. `--allow-empty` opts into the older behavior when a
capability-only probe is intended.
For symbolic grammars without a finite enumerated candidate matrix,
`library_size` is left empty in the `native` protocol.

Some datasets can define equivalent coefficient forms in
`TRUE_COEFFICIENT_ALTERNATIVES`. Metrics choose the structurally closest
accepted form, then the one with the smallest coefficient error. This accepted
form logic is used by clean, noise sweep, and boundary metrics.

Noisy runs report:

- `success_count`: how many of `runs` recovered an accepted true structure;

Boundary noise metrics from `noise_boundary_metrics.py` report:

- `correct_count`: how many noisy runs recovered the expected structure;
- `hd_mean` and `hd_std`: mean and standard deviation of the Hamming distance
  over all completed noisy runs;
- `re_mean` and `re_std`: mean and standard deviation of the relative
  coefficient error, computed only for structurally correct runs;

Here HD counts structural mistakes, for example missing or extra active terms.
RE is separated from HD because coefficient comparison is meaningful only when
the discovered equation has the correct structure.


## Current Results

The current clean and noisy summaries are collected in [`results.md`](results.md).


### Modern EPDE checkout

The shared clean/noise/boundary drivers use `epde/run_config_api.py`.
Select a modern EPDE checkout containing `projects/pic/epde_bench` explicitly:

```sh
export EPDE_PATH=/absolute/path/to/EPDE
python clean_run_metrics.py epde --protocol native --datasets ode_data.npy lotka_data.npy ac_data.npy --algorithm-seed 0
python clean_run_metrics.py epde --protocol fixed --datasets ode_data.npy lotka_data.npy ac_data.npy --epde-path "$EPDE_PATH" --algorithm-seed 0
```

A clean run writes a JSON sidecar beside its common CSV, retaining raw equations,
coefficients, grouped configuration, seed, derivative source, EPDE checkout and
Git revision. The revision identifies the committed base; local checkout changes
must also be preserved with the experiment. The legacy submodule is not upgraded
or initialized by the driver. Invalid or conflicting checkout paths fail explicitly.
The EPDE Docker service mounts the host `EPDE_PATH` at `/opt/epde`; set this
variable before `docker compose` or `run_benchmark.py`. Docker has not been
validated as part of the local smoke checks.

The fixed protocol retains PDEComp's shared numerical pure derivatives in EPDE's
axis-major layout, dataset windows, configured stored features and search budgets.
The native protocol retains EPDE preprocessing and symbolic families without a
shared fixed candidate matrix. Trigonometric frequencies are explicit metadata:
`ode_data.npy` uses exactly 2 in both protocols; native scalar tasks otherwise
use exactly 1; fixed tasks use exactly 2 except `ODE_simple_discovery` (1).
No interval-width heuristic determines whether a family is fixed. Systems retain
their configured variables and have no native trigonometric family. The driver
seeds Python, NumPy and PyTorch independently of the noise realization seed.

EPDE structural/coefficient metrics use the modern `epde_bench.metrics` and
registered accepted truth systems. Registry data files must therefore be present
in the selected checkout even though search data still come from PDEComp's own
`data/` and retain its original windows (including LV's first 150 samples).
The `pde_divide_data.npy` truth explicitly retains PDEComp's -1 and 0.25
coefficients rather than the modern record's -2 and 0.5. Existing accepted
similarity/soliton forms remain accepted; they should not be interpreted as
recovery of a general governing law. Clean smoke completion means the command,
raw result and common metrics agree; it does not promise structural recovery.
The full external dataset campaign and nonzero-noise experiments remain separate
from the checked local smoke scenarios.
