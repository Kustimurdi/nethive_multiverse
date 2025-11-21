#!/usr/bin/env julia
"""
Aggregate `summaries.csv` across replicates and plot results.

Usage (examples):
  julia --project=./env_mutltiverse_analysis aggregate_and_plot.jl <sweep_dir> --metric run_quality_score --x replicate_id
  julia --project=./env_mutltiverse_analysis aggregate_and_plot.jl <sweep_dir> --metric run_quality_score --x learning_rate --y param_id --fix punish_rate=0.01,lambda_sensitivity=1

Behavior:
 - Reads `<sweep_dir>/summaries.csv` produced by `save_log.jl`.
 - Filters rows by `--fix` parameter pairs (key=value, comma-separated).
 - Groups by `x` (and optionally `y`) and computes mean/std/count of `--metric`.
 - If `y` is provided, attempts to render a heatmap (requires CairoMakie).
 - If only `x` is provided, renders a line plot (requires CairoMakie).
 - If CairoMakie is unavailable, writes aggregated CSV to `<sweep_dir>/aggregated_<metric>_<x>[_vs_<y>].csv`.

This script avoids adding new package dependencies; plotting is optional.
"""

using CSV, DataFrames, JSON3, Statistics
using Dates

function parse_keyvals(s::String)
    out = Dict{String,String}()
    for part in split(s, ',')
        if isempty(strip(part))
            continue
        end
        kv = split(part, '=')
        if length(kv) != 2
            error("Invalid key=val pair: $part")
        end
        out[strip(kv[1])] = strip(kv[2])
    end
    return out
end

function coerce_filter(df::DataFrame, key::String, valstr::String)
    # If key not a column, error
    if !(Symbol(key) in names(df))
        error("Column '$key' not found in summaries.csv")
    end
    col = df[!, Symbol(key)]
    # Try to parse valstr to the column eltype
    T = eltype(col)
    parsed = try
        if T <: Integer
            parse(Int, valstr)
        elseif T <: AbstractFloat
            parse(Float64, valstr)
        elseif T <: Bool
            lowercase(valstr) in ("true","1","t")
        elseif T <: Dates.AbstractDateTime
            DateTime(valstr)
        else
            # compare as string for other types (including JSON-stringified arrays)
            valstr
        end
    catch
        # fallback to string comparison
        valstr
    end
    return parsed
end

function aggregate(df::DataFrame, metric::String, x::String, y::Union{String,Nothing}=nothing)
    if !(Symbol(metric) in names(df))
        error("Metric column '$metric' not found in summaries.csv")
    end
    if !(Symbol(x) in names(df))
        error("X column '$x' not found in summaries.csv")
    end
    if !isnothing(y) && !(Symbol(y) in names(df))
        error("Y column '$y' not found in summaries.csv")
    end

    if isnothing(y)
        g = groupby(df, Symbol(x))
        agg = combine(g, Symbol(metric) => mean => :mean, Symbol(metric) => std => :std, Symbol(metric) => length => :count)
        sort!(agg, Symbol(x))
        return agg
    else
        g = groupby(df, [Symbol(x), Symbol(y)])
        agg = combine(g, Symbol(metric) => mean => :mean, Symbol(metric) => std => :std, Symbol(metric) => length => :count)
        return agg
    end
end

function pivot_for_heatmap(agg::DataFrame, x::String, y::String)
    # pivot mean values into matrix with rows=x, cols=y
    if !(:mean in names(agg))
        error("Aggregated DataFrame does not contain :mean column")
    end
    pd = unstack(agg, Symbol(y), :mean)
    # Ensure rows sorted by x
    sort!(pd, Symbol(x))
    # Extract unique x and y ordering
    xs = pd[!, Symbol(x)]
    # columns except x are the y values
    ycols = setdiff(names(pd), [Symbol(x)])
    mat = Matrix(pd[:, ycols])'
    # Return x labels, y labels, matrix (y x x) so heatmap axes correspond
    return xs, Symbol.(ycols), mat
end

function try_plot_line(agg::DataFrame, x::String, metric::String, outpath::String)
    try
        @eval begin
            using CairoMakie
        end
    catch e
        return false, "CairoMakie not available: $e"
    end
    xs = agg[!, Symbol(x)]
    means = agg[!, :mean]
    stds = haskey(names(agg), :std) ? agg[!, :std] : fill(0.0, nrow(agg))
    fig = Figure(resolution=(900,600))
    ax = Axis(fig[1,1], xlabel=x, ylabel=metric, title="$metric vs $x")
    lines!(ax, xs, means, color=:blue)
    scatter!(ax, xs, means, color=:blue)
    # add errorbars if available
    for (xi, m, s) in zip(xs, means, stds)
        lines!(ax, [xi, xi], [m-s, m+s], color=:gray)
    end
    save(outpath, fig)
    return true, "Wrote plot to $outpath"
end

function try_plot_heatmap(agg::DataFrame, x::String, y::String, metric::String, outpath::String)
    ok = try
        @eval begin
            using CairoMakie
        end
        true
    catch e
        false
    end
    if !ok
        return false, "CairoMakie not available"
    end
    xs, ylabels, mat = pivot_for_heatmap(agg, x, y)
    fig = Figure(resolution=(1000,800))
    ax = Axis(fig[1,1])
    heatmap!(ax, reverse(mat,dims=1), colormap=:viridis)
    ax.yticks = (1:length(ylabels), string.(reverse(ylabels)))
    ax.xticks = (1:length(xs), string.(xs))
    Colorbar(fig[1,2], ax; label=metric)
    fig[1,1] .= ax
    save(outpath, fig)
    return true, "Wrote heatmap to $outpath"
end

function main()
    if length(ARGS) < 1
        println("Usage: aggregate_and_plot.jl <sweep_dir> --metric <metric> --x <xcol> [--y <ycol>] [--fix k=v,k2=v2] [--out out.png]")
        return
    end
    sweep_dir = ARGS[1]
    # parse remaining args
    metric = "run_quality_score"
    x = nothing
    y = nothing
    fix = Dict{String,String}()
    out = nothing
    i = 2
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--metric"
            i += 1; metric = ARGS[i]
        elseif a == "--x"
            i += 1; x = ARGS[i]
        elseif a == "--y"
            i += 1; y = ARGS[i]
        elseif a == "--fix"
            i += 1; fix = parse_keyvals(ARGS[i])
        elseif a == "--out"
            i += 1; out = ARGS[i]
        else
            error("Unknown arg: $a")
        end
        i += 1
    end
    if x === nothing
        error("--x is required (the parameter to aggregate along)")
    end

    csvp = joinpath(sweep_dir, "summaries.csv")
    if !isfile(csvp)
        error("summaries.csv not found in $sweep_dir")
    end

    df = CSV.read(csvp, DataFrame)

    # Apply fixes
    for (k,v) in fix
        parsed = coerce_filter(df, k, v)
        df = filter(row -> begin
            val = row[Symbol(k)]
            # Compare by string if types mismatch
            if typeof(val) != typeof(parsed)
                string(val) == string(parsed)
            else
                val == parsed
            end
        end, df)
    end

    agg = aggregate(df, metric, x, y)

    # Default output path
    if out === nothing
        out = joinpath(sweep_dir, "aggregated_$(metric)_$(x)" * (isnothing(y) ? "" : "_vs_$(y)") * ".png")
    end

    if isnothing(y)
        # 1D plot
        ok, msg = try_plot_line(agg, x, metric, out)
        if ok
            println(msg)
        else
            # write CSV instead
            outcsv = replace(out, r"\.png$" => ".csv")
            CSV.write(outcsv, agg)
            println("Could not plot (", msg, "). Wrote aggregated CSV to: ", outcsv)
        end
    else
        ok, msg = try_plot_heatmap(agg, x, y, metric, out)
        if ok
            println(msg)
        else
            outcsv = replace(out, r"\.png$" => ".csv")
            CSV.write(outcsv, agg)
            println("Could not plot (", msg, "). Wrote aggregated CSV to: ", outcsv)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
