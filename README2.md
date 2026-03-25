# nethive_multiverse

Julia code for simulating an **ensemble of interacting neural agents** that train on multiple tasks while **suppressing** one another’s task-specific training for finite refractory times.

Main components:
- Simulation engine (`run_simulation.jl` + `src/`)
- HPC/SLURM sweep manager (`slurm_manager.jl`)
- Synthetic Gaussian task generation (`gauss_generator.jl` + `gauss_datasets/`)

This README is intended for someone who wants to (1) run a single simulation, and/or (2) generate and run parameter sweeps on a SLURM cluster.

---

## Repository layout

```
├── env_nethive_multiverse/ # Julia project environment (Project.toml/Manifest.toml)
├── src/ # core simulation code (model, events, training, IO)
├── run_simulation.jl # run ONE simulation (from config or CLI args)
├── slurm_manager.jl # generate per-run configs + SLURM scripts for sweeps
├── task_loading.jl # task/dataset loading utilities (used by run_simulation)
├── gauss_generator.jl # generate synthetic Gaussian task datasets
├── gauss_datasets/ # stored synthetic task datasets (.jld2)
├── multiverse_analysis/ # analysis scripts (may be messy / not “productized”)
├── special_loading.jl # helpers for special loading/continuation cases
└── warmup.slurm / warmup_v2.sl # optional cluster warmup scripts (precompile etc.)
```

---

## Installation / environment

This project uses a dedicated Julia environment in `env_nethive_multiverse/`.

From the repo root:

```bash
julia --project=./env_nethive_multiverse -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

---

## Running a single simulation (`run_simulation.jl`)

`run_simulation.jl` runs **one** simulation. It supports two modes:
- **Config-driven** (recommended): `--config path/to/config.json`
- **CLI-driven**: specify parameters directly (overrides config file values)

### Minimal config-driven run

```bash
julia --project=./env_nethive_multiverse run_simulation.jl \
  --config path/to/config.json \
  --output-dir results/ \
  --base-name my_run \
  --save-results
```

### Minimal CLI-driven run (example)

```bash
julia --project=./env_nethive_multiverse run_simulation.jl \
  --n-bees 10 \
  --dataset-names task_1 task_2 task_3 task_4 task_5 task_6 task_7 task_8 task_9 task_10 \
  --n-epochs 10000 \
  --production-rate 1.0 \
  --interaction-rate 200.0 \
  --dead-time 100.0 \
  --lambda-sensitivity 1e6 \
  --punishment time_out \
  --save-results
```

### Output location
By default, `run_simulation.jl` writes a new run folder inside `--output-dir`. The folder name is based on `--base-name` (and includes a timestamp if `--timestamp` is set). Each run folder contains `config.json`, `states.csv`, `activity.csv`, and `task_index_mapping.json` (plus optional checkpoints such as `epoch_<k>/`).

### CLI reference

```bash
julia --project=./env_nethive_multiverse run_simulation.jl --help
```

---

## Running parameter sweeps on SLURM (`slurm_manager.jl`)

`slurm_manager.jl` generates a sweep folder that contains:
- one **per-run JSON config** per parameter combination and replicate
- a SLURM job script (job array if requested)
- standard subfolders for outputs and logs

### Typical usage (job array; recommended)

Generate a sweep folder:

```bash
julia --project=./env_nethive_multiverse slurm_manager.jl generate \
  -c path/to/sweep_config.json \
  -o checkpoint/my_sweep \
  -f checkpoint \
  --replicates 5 \
  --use-array \
  -p th-ws \
  -t 50:00:00 \
  -m 3 \
  --cpus 1
```

This creates (example):

```
checkpoint/my_sweep/
├── configs/            # per-run config.json files (one per run)
├── data/               # run outputs: param_*_rep_*/
├── logs/               # SLURM stdout/stderr
├── scripts/            # helper scripts
├── job_array.slurm     # array job definition
└── submit_array.sh     # convenience submit wrapper
```

Submit (inside the sweep folder):

```bash
bash submit_array.sh
```

Note: `slurm_manager.jl` can also generate scripts for submitting runs individually (non-array), but job arrays are usually preferred on the cluster and are what I used in practice.

### CLI reference

```bash
julia --project=./env_nethive_multiverse slurm_manager.jl --help
``` 

Notable options:

- `--use-array` : generate an array job instead of individual jobs
- resources: `--partition`, `--time`, `--memory`, `--cpus`, `--array-max`
- reproducibility: `--replicates`, `--base-seed`

Other options exist (e.g. depot isolation / precompile flags), but I did not rely on them heavily and they may be untested in my workflow.

---

## Sweep-config JSON schema (for `slurm_manager.jl`)

A sweep config file has two parts:
- `base_config`: defaults used for every run
- `sweep_parameters`: parameters that are varied; each entry is a list of values

`slurm_manager.jl` will create **one run for every combination** of the values listed in `sweep_parameters`, and then repeat that for the requested number of replicates.

### Example sweep config

```json
{
  "base_config": {
    "n_bees": 2,
    "n_tasks": 2,
    "dataset_names": ["bank", "wdbc"],
    "n_epochs": 1000,
    "n_steps_per_epoch": 1,
    "batch_size": 64,
    "save_nn_epochs": 10000,
    "seed": null,

    "punish_rate": 0.1,
    "production_rate": 1.0,
    "punishment": "time_out",

    "n_classes": 15,
    "features_dimension": 10,
    "n_per_class_train": 100,
    "n_per_class_test": 50,
    "use_gauss": true,
    "use_per_class_variance": true,
    "variance_bounds": [0.01, 0.05],
    "batches_per_step": 1000,
    "center_generation_bounds": [0.0, 1.0]
  },

  "sweep_parameters": {
    "lambda_sensitivity": [100.0],
    "learning_rate": [0.00005],
    "interaction_rate": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    "dead_time": [0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]
  }
}
```

### How fields are interpreted

#### `base_config`
Everything in `base_config` is copied into every per-run config unless overwritten by a sweep value.

Common entries:
- **System size**
  - `n_bees`: number of agents
  - `n_tasks`: number of tasks

- **Run length / logging**
  - `n_epochs`: number of recorded epochs 
  - `n_steps_per_epoch`: legacy parameter (older stepping logic; in most recent runs, `n_steps_per_epoch` is not used and may be set to `-1`)
  - `save_nn_epochs`: checkpoint interval in epochs. When enabled, saves per-agent network parameters and suppression state (including remaining refractory times) into `epoch_<k>/` folders. When set to `0`, checkpoint saving is disabled.

- **Training**
  - `learning_rate`: optimizer step size
  - `batch_size`: minibatch size
  - `batches_per_step`: number of minibatches per training event

- **Event rates / interaction parameters**
  - `production_rate`: baseline training event rate (λ_tr)
  - `interaction_rate`: suppression-rate prefactor (α)
  - `dead_time`: refractory duration (τ)
  - `lambda_sensitivity`: sigmoid steepness / interaction asymmetry (η)
  - `punishment`: suppression mode (typically `time_out`; valid values: `time_out`, `resetting`, `gradient_ascend`, `none`)
  - `punish_rate`: legacy parameter (used by `gradient_ascend`)

- **Task / dataset selection**
  - `dataset_names`: list of task identifiers (strings)
  - `use_gauss`: generate Gaussian tasks on the fly from parameters
  - `use_gauss_dataset`: load a pre-generated Gaussian dataset file (requires `gauss_dataset_dir` + `gauss_dataset_name`)

Gaussian generation parameters (when `use_gauss=true`) include:
- `n_classes`
- `features_dimension`
- `n_per_class_train`, `n_per_class_test`
- `center_generation_bounds`
- `use_per_class_variance`, `variance_bounds`

#### `sweep_parameters`
Each key must match a field that `run_simulation.jl` understands (e.g. `interaction_rate`, `dead_time`, `lambda_sensitivity`, `learning_rate`). The sweep will run all combinations of those lists.

In the example above:
- `lambda_sensitivity` and `learning_rate` are fixed (single value lists)
- `interaction_rate` has 10 values and `dead_time` has 21 values → 210 parameter points, then multiplied by the number of replicates.

### Replicates and seeds
- `--replicates R` creates `R` runs per parameter point.
- `base_config.seed` can be `null` so the manager assigns seeds automatically.
- `--base-seed` can be used to make replicate seeds reproducible when regenerating the sweep.

---

## Output folder naming (`param_*_rep_*`)

Each run is stored as:
`data/param_<param_id>_rep_<replicate_id>/`

- `param_id`: which parameter combination in the sweep
- `replicate_id`: replicate index / seed instance for that parameter point

Each run folder contains at minimum:
- `config.json`
- `states.csv`
- `activity.csv`
- `task_index_mapping.json`

Optionally (depending on run type / continuation / saving settings):
- `epoch_<k>/` checkpoint folders (per-agent model snapshots + suppression state)
- `loaded_bee_mapping.jdl2` (for continuation runs that remap agents)

---

## Outputs of a run (data format)

### `config.json`
Full configuration used for the run (parameters + metadata like timestamps / git commit).

### `states.csv`
Columns:
- `epoch::Int`
- `bee_id::Int`
- `task_id::Int`
- `accuracies::Float64`

Notes:
- `epoch == 0` records accuracies at initialization (before the first Gillespie event).

### `activity.csv`
Columns:
- `epoch::Int`
- `bee_id::Int`
- `task_id::Int`
- `suppressed::Bool` (true = suppressed/inactive on that task)

### `task_index_mapping.json`
Mapping from internal integer task IDs to task names (strings). Example:
```json
{
  "1": "task_1",
  "2": "task_2",
  "11": "task_48"
}
```

---

## Synthetic Gaussian tasks (`gauss_generator.jl`, `gauss_datasets/`)

There are two ways Gaussian tasks appear in runs:
- `use_gauss=true`: tasks are generated from config parameters at runtime
- `use_gauss_dataset=true`: tasks are loaded from a stored `.jld2` dataset file 

To generate new stored datasets (if you use the stored-dataset workflow):
```bash
julia --project=./env_nethive_multiverse gauss_generator.jl
```

If you want different Gaussian dataset parameters, edit `gauss_generator.jl`.

---

## Analysis (optional)

`multiverse_analysis/` contains scripts used to aggregate and plot results (heatmaps, radar plots, ECDFs, etc.). The code there is functional but not organized as a clean API; for “reproduce thesis plots” workflows, prefer plot scripts that directly export the relevant figures.

Typical analysis workflow:
1. collect run outputs from `.../data/param_*_rep_*/`
2. compute per-run summary metrics (coverage/specialization, onset times, activity diagnostics)
3. aggregate across replicates (mean/std)
4. plot