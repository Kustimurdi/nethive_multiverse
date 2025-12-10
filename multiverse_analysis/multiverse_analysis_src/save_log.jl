#!/usr/bin/env julia
# Save run summaries (compact CSV + detailed JSONL) for a sweep of runs.
# Places outputs into the sweep directory. Primary input: event_log.csv in each run folder.
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_analysis/env_mutltiverse_analysis")

using DataFrames
using CSV
using JSON3
using Dates

#const DEFAULT_SWEEP = "/project/theorie/n/N.Pfaffenzeller/multiverse_2/gauss_runs/10b10t10c_high_deadtimes/"
#const DEFAULT_SWEEP = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_runs/4b4t20c_testrun"
#const DEFAULT_SWEEP = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_runs/4b4t10c/"
#const DEFAULT_SWEEP = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_runs/real_time/15b15t20c100e/"
#const DEFAULT_SWEEP = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_runs/real_time/10b10t20c100e/"
const DEFAULT_SWEEP = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_runs/real_time/15b15t20c200e"

include(joinpath(@__DIR__, "load_data.jl"))
include(joinpath(@__DIR__, "analyse_log.jl"))

function sanitize_for_json(x)
    # Convert common Julia objects into JSON-friendly representations
    if x === nothing
        return nothing
    elseif x isa Missing
        return nothing
    elseif x isa Float64 && isnan(x)
        return nothing
    elseif x isa NamedTuple
        d = Dict{String,Any}()
        for (k,v) in pairs(x)
            d[string(k)] = sanitize_for_json(v)
        end
        return d
    elseif x isa Dict
        d = Dict{String,Any}()
        for (k,v) in pairs(x)
            d[string(k)] = sanitize_for_json(v)
        end
        return d
    elseif x isa AbstractVector
        return [sanitize_for_json(v) for v in x]
    else
        # Try to convert objects that expose propertynames (e.g. DataFrameRow)
        try
            d = Dict{String,Any}()
            for n in propertynames(x)
                try
                    v = getfield(x, n)
                catch
                    v = x[n]
                end
                d[string(n)] = sanitize_for_json(v)
            end
            return d
        catch
            return x
        end
    end
end

# Return a CSV-safe scalar representation for arbitrary values (no newlines)
function safe_scalar(x)
    if x === nothing || x isa Missing
        return ""
    elseif isa(x, Number) || isa(x, Bool)
        return x
    elseif isa(x, AbstractString)
        return replace(x, '\n' => ' ')
    else
        # Use compact JSON representation and strip newlines
        try
            s = JSON3.write(x)
            return replace(s, '\n' => ' ')
        catch
            return replace(string(x), '\n' => ' ')
        end
    end
end

function process_sweep(sweep_dir::String; out_csv_name::String="summaries.csv", out_jsonl_name::String="summaries.jsonl", resume::Bool=true, max_runs::Int=0)
    println("Scanning sweep directory: ", sweep_dir)
    if !isdir(sweep_dir)
        error("sweep directory not found: $sweep_dir")
    end

    # Most sweeps store run folders under a 'data' subdirectory. If present,
    # scan that directory for runs. Otherwise, fall back to scanning the
    # sweep_dir itself.
    data_dir = joinpath(sweep_dir, "data")
    run_parent = isdir(data_dir) ? data_dir : sweep_dir

    out_csv = joinpath(sweep_dir, out_csv_name)
    out_jsonl = joinpath(sweep_dir, out_jsonl_name)
    existing_runs = Set{String}()
    if resume && isfile(out_csv)
        try
            df_existing = CSV.read(out_csv, DataFrame)
            if :run_path in names(df_existing)
                for p in df_existing[!, :run_path]
                    push!(existing_runs, String(p))
                end
            end
        catch e
            @warn "Could not read existing CSV: $e. Will overwrite.";
        end
    end

    # Prepare CSV rows
    rows = DataFrame()
    first_write = !isfile(out_csv)

    jsonl_io = open(out_jsonl, resume && isfile(out_jsonl) ? "a" : "w")

    try
        processed = 0
        for entry in sort(readdir(run_parent))
            run_path = joinpath(run_parent, entry)
            if !isdir(run_path)
                continue
            end
            log_file = joinpath(run_path, "event_log.csv")
            if !isfile(log_file)
                # also accept event_log.json (but primary is CSV)
                continue
            end
            if String(run_path) in existing_runs
                println("Skipping already-processed run: ", run_path)
                continue
            end

            println("Processing run: ", run_path)
            try
                log = load_log(run_path)
                summary = run_summary(log)

                # Build compact row and merge with run config (flattened)
                row = Dict(
                    "run_path" => run_path,
                    "run_quality_score" => summary.run_quality_score,
                    "p_end_mean" => summary.run_score_mean_best_task_at_end,
                    "coverage_fraction" => summary.coverage_fraction,
                    "assignment_entropy_totalbees" => summary.assignment_entropy_totalbees,
                    "run_score_mean_best_task" => summary.run_score_mean_best_task,
                    "n_events" => summary.n_events,
                    "n_training" => summary.n_training,
                    "duration" => summary.duration,
                    "n_bees" => summary.n_bees,
                    "n_tasks" => summary.n_tasks,
                    "mean_gain_per_training" => summary.mean_gain_per_training,
                    "run_mean_task_value_at_interaction" => summary.run_mean_task_value_at_interaction,
                    "specialization_score_mean" => summary.specialization_score_mean,
                    "run_quality_components" => safe_scalar(summary.run_quality_components)
                )

                # Load and flatten config.json into additional columns (if present)
                try
                    cfg_df = load_config(joinpath(run_path, "config.json"))
                    if nrow(cfg_df) >= 1
                        cfg_row = Dict{String,Any}()
                        for cname in names(cfg_df)
                            # convert symbol names to strings
                            s = string(cname)
                            cfg_row[s] = safe_scalar(cfg_df[1, cname])
                        end
                        # Merge: config keys first, row values override if collisions
                        row = merge(cfg_row, row)
                    end
                catch e
                    @warn "Could not load config for $run_path: $e"
                end

                # Append to CSV (write header if needed)
                dfrow = DataFrame(row)
                if first_write
                    CSV.write(out_csv, dfrow)
                    first_write = false
                else
                    CSV.write(out_csv, dfrow; append=true)
                end

                # Append detailed JSONL (full sanitized summary)
                obj = Dict("run_path" => run_path, "summary" => sanitize_for_json(summary))
                JSON3.write(jsonl_io, obj)
                write(jsonl_io, '\n')

                processed += 1
                if max_runs > 0 && processed >= max_runs
                    println("Reached max_runs limit ($max_runs). Stopping early.")
                    break
                end

            catch e
                @warn "Error processing $run_path: $e"
                # write an error entry to JSONL
                obj = Dict("run_path" => run_path, "error" => string(e))
                JSON3.write(jsonl_io, obj)
                write(jsonl_io, '\n')
            end
        end
    finally
        close(jsonl_io)
    end

    println("Done. CSV written to: $out_csv")
    println("JSONL written to: $out_jsonl")
    return (csv=out_csv, jsonl=out_jsonl)
end

function main()
    sweep_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_SWEEP
    max_runs = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
    t0 = now()
    res = process_sweep(sweep_dir; max_runs=max_runs)
    t1 = now()
    println("Elapsed: ", t1 - t0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
