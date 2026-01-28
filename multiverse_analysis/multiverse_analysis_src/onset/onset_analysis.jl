using DataFrames, CSV
using JSON3

# ----------------------------
# Helpers: discovery
# ----------------------------

function find_param_dirs(root::AbstractString)
    data_dir = joinpath(root, "data")
    isdir(data_dir) || return String[]
    dirs = String[]
    for name in readdir(data_dir)
        full = joinpath(data_dir, name)
        if isdir(full) && startswith(name, "param_")
            push!(dirs, full)
        end
    end
    sort!(dirs)
    return dirs
end

"Return a vector of NamedTuples describing runs to analyze."
function collect_runs(roots::Vector{String};
    event_filename::AbstractString = "event_log.csv",
    config_filename::AbstractString = "config.json",
)
    runs = NamedTuple[]
    for root in roots
        for param_dir in find_param_dirs(root)
            param_folder = splitdir(param_dir)[2]
            event_log_path  = joinpath(param_dir, event_filename)
            config_path = joinpath(param_dir, config_filename)
            push!(runs, (
                root = root,
                param_dir = param_dir,
                param_folder = param_folder,
                event_log_path = event_log_path,
                config_path = config_path,
            ))
        end
    end
    return runs
end

# ----------------------------
# Helpers: config.json -> DataFrame
# ----------------------------

"""
Flatten a JSON object into a Dict with dot-separated keys.
- nested objects become key.subkey
- arrays become JSON-encoded strings (keeps it simple + CSV-friendly)
"""
function flatten_json(x; prefix::String = "", out::Dict{String,Any} = Dict{String,Any}())
    if x isa JSON3.Object
        for (k, v) in pairs(x)
            key = isempty(prefix) ? String(k) : string(prefix, ".", k)
            flatten_json(v; prefix = key, out = out)
        end
    elseif x isa JSON3.Array
        out[prefix] = JSON3.write(x)  # store array as a JSON string
    else
        out[prefix] = x
    end
    return out
end

function config_to_df(config_path::AbstractString)
    obj = JSON3.read(read(config_path, String))
    flat = flatten_json(obj)
    df = DataFrame(flat)
    return df
end

# ----------------------------
# Main per-run loader
# ----------------------------

"""
Analyze one run folder:
- config.json provides base columns
- analysis adds onset/specialization columns
Returns a 1-row DataFrame.
"""
function analyze_one_run(run; thresholds = [0.8, 0.9])
    # sanity checks
    isfile(run.config_path) || error("missing config.json")
    isfile(run.event_log_path) || error("missing event_log.csv")

    cfg = config_to_df(run.config_path)                 # 1-row DF
    ana = analyze_event_log_csv(run.event_log_path; thresholds=thresholds)  # 1-row DF (your function)

    # merge (avoid column collisions)
    row = hcat(cfg, ana; makeunique=true)

    # add metadata
    row[!, :root] = [run.root]
    row[!, :param_folder] = [run.param_folder]
    row[!, :param_dir] = [run.param_dir]
    row[!, :event_log_path] = [run.event_log_path]
    row[!, :config_path] = [run.config_path]

    return row
end

# ----------------------------
# Batch runner (parallel)
# ----------------------------

"""
Analyze all runs in parallel using threads.
Returns:
- results::DataFrame
- failures::DataFrame (with error messages)
"""
function analyze_all_runs_from_config(roots::Vector{String};
    thresholds = [0.8, 0.9],
    event_filename = "event_log.csv",
    config_filename = "config.json",
)
    runs = collect_runs(roots; event_filename=event_filename, config_filename=config_filename)
    n = length(runs)

    results_vec = Vector{Union{Nothing,DataFrame}}(nothing, n)
    failures = Vector{NamedTuple}(undef, 0)
    fail_lock = ReentrantLock()

    Threads.@threads for i in 1:n
        run = runs[i]
        try
            results_vec[i] = analyze_one_run(run; thresholds=thresholds)
        catch e
            lock(fail_lock) do
                push!(failures, (
                    root = run.root,
                    param_folder = run.param_folder,
                    param_dir = run.param_dir,
                    event_log_path = run.event_log_path,
                    config_path = run.config_path,
                    error = sprint(showerror, e),
                ))
            end
        end
    end

    rows = [df for df in results_vec if df !== nothing]
    results = isempty(rows) ? DataFrame() : vcat(rows...; cols=:union)
    failures_df = isempty(failures) ? DataFrame() : DataFrame(failures)

    return results, failures_df
end

function save_analysis_outputs(results::DataFrame, failures::DataFrame, outdir::AbstractString;
    prefix::AbstractString = "summary",
)
    mkpath(outdir)
    results_csv  = joinpath(outdir, "$(prefix).csv")
    failures_csv = joinpath(outdir, "$(prefix)_failures.csv")
    CSV.write(results_csv, results)
    CSV.write(failures_csv, failures)
    return (results_csv=results_csv, failures_csv=failures_csv)
end
