"""
Data Loading Module for NetHive Multiverse Analysis

Expected folder structure for simulation runs:
```
base_dir/
├── param_1_rep_1_timestamp/
│   ├── config.json         # Run configuration and metadata
│   ├── states.csv          # Performance history (epoch, bee_id, task_id, accuracies)
│   ├── events.csv          # Event counts (time_step, time, bee_id, task_id, production_count, suppression_count)
│   └── losses.csv          # Loss history (optional)
├── param_2_rep_1_timestamp/
│   └── ...
└── ...
```

Configuration files can be either:
- Flat format (from save_metadata_to_config): direct key-value pairs
- Nested format (from Slurm manager): {"base_config": {...}, "sweep_parameters": {...}}

Before using, activate the analysis environment:
```julia
using Pkg; Pkg.activate("multiverse_analysis/env_mutltiverse_analysis")
```
"""

using Pkg
# Removed hard-coded activation - activate env_mutltiverse_analysis before running
using CSV
using DataFrames
using JSON3

function load_config(config_path::String)
    # Load and parse JSON configuration
    txt = read(config_path, String)
    obj = JSON3.read(txt)
    
    # Handle both flat configs (from save_metadata_to_config) and nested Slurm configs
    if isa(obj, JSON3.Object) && haskey(obj, "base_config")
        obj = obj["base_config"]  # Use nested config from Slurm manager
    end
    
    # Ensure that any non-scalar (arrays, objects) are converted to compact JSON
    # strings so the resulting DataFrame is a single-row table rather than
    # expanding array-valued fields into multiple rows.
    function scalarize_config_value(v)
        try
            if isa(v, AbstractVector) || isa(v, Dict) || isa(v, JSON3.Array) || isa(v, JSON3.Object)
                return JSON3.write(v)
            else
                return v
            end
        catch
            return v
        end
    end

    row = Dict(Symbol(k) => scalarize_config_value(v) for (k, v) in pairs(obj))

    cfg_df = DataFrame([row])
    return cfg_df
end

# row = Dict(Symbol(k) => v for (k, v) in pairs(obj))

function load_states(states_path::String)
    # Load states CSV and normalize column names for analysis compatibility
    df = CSV.read(states_path, DataFrame)
    
    # Normalize column names to what analysis functions expect
    if hasproperty(df, :accuracies)
        rename!(df, :accuracies => :task_value)
    end
    
    # Ensure required columns exist
    required_cols = [:epoch, :bee_id, :task_id, :task_value]
    for col in required_cols
        if !hasproperty(df, col)
            error("States CSV missing required column: $col")
        end
    end
    
    return df
end

function load_events(events_path::String)
    # Load events CSV
    if !isfile(events_path)
        @warn "Events file not found: $events_path. Returning empty DataFrame."
        return DataFrame()
    end
    events = CSV.read(events_path, DataFrame)
    return events
end

function load_log(run_path::String)
    # Load losses CSV (optional file)
    log_path = joinpath(run_path, "event_log.csv")
    if !isfile(log_path)
        @warn "Log file not found: $log_path. Returning empty DataFrame."
        return DataFrame()
    end
    log = CSV.read(log_path, DataFrame)
    return log
end

function load_run(run_path::String; load_all::Bool=false)
    # Load all components of a simulation run
    cfg = load_config(joinpath(run_path, "config.json"))
    states = load_states(joinpath(run_path, "states.csv"))
    events = load_events(joinpath(run_path, "events.csv"))
    if load_all
        log = load_log(joinpath(run_path, "event_log.csv"))
        return (run_path=run_path, config=cfg, states=states, events=events, log=log)
    end
    
    return (run_path=run_path, config=cfg, states=states, events=events)
end

function build_master_dataframe(base_dir::String)
    rows = DataFrame[]
    
    for entry in readdir(base_dir; join=true)
        config_file = joinpath(entry, "config.json")
        if isfile(config_file)
            cfg = load_config(config_file)
            cfg[!, :run_path] = fill(entry, nrow(cfg))  # Safer than broadcasting assignment
            push!(rows, cfg)
        end
    end
    
    if isempty(rows)
        @warn "No valid run directories found in $base_dir"
        return DataFrame()
    end
    
    master_df = vcat(rows...)
    return master_df
end

function load_runs(runs_paths::Vector{String}; only_states::Bool=false)
    if only_states
        # Load only states for multiple simulation runs
        runs = [load_states(joinpath(run_path, "states.csv")) for run_path in runs_paths]
        return runs
    end
    # Load multiple simulation runs
    runs = Vector{Any}(undef, length(runs_paths))
    Threads.@threads for i in eachindex(runs_paths)
        runs[i] = load_run(runs_paths[i])
    end
    return runs
end

function load_processed_run(run_path::String)
    config = load_config(joinpath(run_path, "config.json"))
    analysis_path = joinpath(run_path, "analysis.csv")
    
    analysis_df = isfile(analysis_path) ? CSV.read(analysis_path, DataFrame) : error("$analysis_path is not a file")
    
    return (analysis_df=analysis_df, run_path=run_path, config=config)
end

function load_processed_runs(runs_paths::Vector{String})
    processed_runs = [load_processed_run(run_path) for run_path in runs_paths]
    return processed_runs
end





#probably not needed - but could be useful for loading specific components, ich kann ja auch einfach die funktionen aufrufen
function load_run_component(run_path::String, component::Symbol)
    # Load a specific component of a simulation run
    if component == :config
        return load_config(joinpath(run_path, "config.json"))
    elseif component == :states
        return load_states(joinpath(run_path, "states.csv"))
    elseif component == :events
        return load_events(joinpath(run_path, "events.csv"))
    elseif component == :timeseries
        return load_timeseries(joinpath(run_path, "timeseries.csv"))
    elseif component == :summary
        return load_summary(joinpath(run_path, "summary.csv"))
    else
        error("Unknown component: $component")
    end
end

function load_task_index_mapping(run_path::String)
    mapping_path = joinpath(run_path, "task_index_mapping.json")
    # Load task index to name mapping from CSV
    if !isfile(mapping_path)
        error("Mapping file not found: $mapping_path")
    end
    txt = read(mapping_path, String)
    obj = JSON3.read(txt)
    mapping = Dict{Any,String}()
    for (k, v) in pairs(obj)
        # Keys coming from JSON may be strings or symbols; normalize safely
        ks = String(k)
        ki = tryparse(Int, ks)
        if ki !== nothing
            mapping[ki] = String(v)
        else
            mapping[ks] = String(v)
        end
    end
    return mapping
end