using DataFrames, Statistics

"""
Compute run-level summary metrics from an event log DataFrame.

Returns a NamedTuple with:
 - run_score_mean_best_task: mean over tasks of the best-per-task value seen (primary score)
 - n_events, n_training, n_suppression
 - mean_task_value_at_interaction: average task value when an interaction happened (for the task involved)
 - mean_gain_per_training: average gain (post - pre) for the trained task on training events (missing if not computable)
 - final_mean_accuracy: mean over tasks of the last-observed best bee accuracy (across bees) at end of run
 - duration, n_bees, n_tasks, per_task_best (Dict), per_task_mean_gain (Dict)
"""
# Helper: extract numeric task ids from column names with given prefix
function extract_task_ids(log::DataFrame, task_prefix::String="task_")::Vector{Int}
    task_cols = filter(c -> startswith(string(c), task_prefix), names(log))
    task_ids = Int[]
    for c in task_cols
        m = match(Regex("^" * task_prefix * raw"(\d+)$"), string(c))
        if m === nothing
            continue
        end
        push!(task_ids, parse(Int, m.captures[1]))
    end
    sort!(task_ids)
    return task_ids
end


function compute_per_task_best(log::DataFrame, task_ids::Vector{Int}; task_prefix::String="task_")
    per_task_best = Dict{Int,Float64}()
    for tid in task_ids
        col = Symbol(task_prefix * string(tid))
        vals = skipmissing(Float64.(log[!, col]))
        per_task_best[tid] = isempty(vals) ? NaN : maximum(vals)
    end
    return per_task_best
end


function compute_per_bee_lastrow(log::DataFrame; bee1col::Symbol=:bee1_id, bee2col::Symbol=:bee2_id, timecol::Symbol=:time)
    bees = unique(vcat(log[!, bee1col], log[!, bee2col]))
    per_bee_lastrow = Dict{Any,DataFrameRow}()
    for b in bees
        rows_actor = log[log[!, bee1col] .== b, :]
        if nrow(rows_actor) > 0
            idx = argmax(rows_actor[!, timecol])
            per_bee_lastrow[b] = rows_actor[idx, :]
        end
    end
    return per_bee_lastrow
end


function compute_per_task_end_best(per_bee_lastrow::Dict{Any,DataFrameRow}, task_ids::Vector{Int}; task_prefix::String="task_")
    per_task_end_best = Dict{Int,Float64}()
    for tid in task_ids
        col = Symbol(task_prefix * string(tid))
        vals = Float64[]
        for (b, brow) in per_bee_lastrow
            if hasproperty(brow, col) || haskey(brow, col)
                v = brow[col]
                if !ismissing(v)
                    push!(vals, float(v))
                end
            end
        end
        per_task_end_best[tid] = isempty(vals) ? NaN : maximum(vals)
    end
    return per_task_end_best
end


function compute_task_values_at_events(log::DataFrame; task_prefix::String="task_", taskidcol::Symbol=:task_id)
    task_vals_at_event = Float64[]
    for row in eachrow(log)
        col = Symbol(task_prefix * string(row[taskidcol]))
        if hasproperty(row, col) || haskey(row, col)
            v = row[col]
            if !ismissing(v)
                push!(task_vals_at_event, float(v))
            end
        end
    end
    mean_task_value_at_interaction = isempty(task_vals_at_event) ? missing : mean(task_vals_at_event)
    return mean_task_value_at_interaction, task_vals_at_event
end


function compute_training_gains(log::DataFrame; task_prefix::String="task_", bee1col::Symbol=:bee1_id,
                                bee2col::Symbol=:bee2_id, taskidcol::Symbol=:task_id, timecol::Symbol=:time)
    gains = Float64[]
    grouped = groupby(log, [bee1col, taskidcol])
    for g in grouped
        gsorted = sort(g, timecol)
        if nrow(gsorted) < 2
            continue
        end
        colname = Symbol(task_prefix * string(gsorted[1, taskidcol]))
        vals = Float64.(gsorted[!, colname])
        for k in 2:length(vals)
            row_k = gsorted[k, :]
            if row_k[bee1col] == row_k[bee2col]
                delta = vals[k] - vals[k-1]
                push!(gains, float(delta))
            end
        end
    end
    mean_gain = isempty(gains) ? missing : mean(gains)
    return gains, mean_gain
end


function compute_per_task_gains(log::DataFrame, task_ids::Vector{Int}; task_prefix::String="task_",
                                bee1col::Symbol=:bee1_id, bee2col::Symbol=:bee2_id, taskidcol::Symbol=:task_id, timecol::Symbol=:time)
    per_task_gains = Dict{Int,Float64}()
    for tid in task_ids
        col = Symbol(task_prefix * string(tid))
        gains_tid = Float64[]
        for g in groupby(log[log[!, taskidcol] .== tid, :], bee1col)
            gsorted = sort(g, timecol)
            if nrow(gsorted) < 2
                continue
            end
            vals = Float64.(gsorted[!, col])
            for k in 2:length(vals)
                row_k = gsorted[k, :]
                if row_k[bee1col] == row_k[bee2col]
                    push!(gains_tid, float(vals[k] - vals[k-1]))
                end
            end
        end
        per_task_gains[tid] = isempty(gains_tid) ? NaN : mean(gains_tid)
    end
    return per_task_gains
end


function compute_specialization_metrics(log::DataFrame, task_ids::Vector{Int}; task_prefix::String="task_",
                                        bee1col::Symbol=:bee1_id, bee2col::Symbol=:bee2_id, timecol::Symbol=:time)
    # Build last-known per-bee state
    last_known = Dict{Any, Dict{Int,Float64}}()
    sorted_log = sort(log, timecol)
    for row in eachrow(sorted_log)
        b = row[bee1col]
        if !haskey(last_known, b)
            last_known[b] = Dict{Int,Float64}()
        end
        for tid in task_ids
            col = Symbol(task_prefix * string(tid))
            if (hasproperty(row, col) || haskey(row, col)) && !ismissing(row[col])
                last_known[b][tid] = float(row[col])
            end
        end
    end
    # ensure presence of all bees
    for b in unique(vcat(log[!, bee1col], log[!, bee2col]))
        if !haskey(last_known, b)
            last_known[b] = Dict{Int,Float64}()
        end
        for tid in task_ids
            if !haskey(last_known[b], tid)
                last_known[b][tid] = NaN
            end
        end
    end

    # compute per-bee and per-task specialization numbers
    per_bee_best_task = Dict{Any,Int}()
    per_bee_best_value = Dict{Any,Float64}()
    per_bee_second_value = Dict{Any,Float64}()
    per_bee_spec_score = Dict{Any,Float64}()
    eps = 1e-9
    for (b, tv) in last_known
        vals = [tv[tid] for tid in task_ids]
        real_idx = findall(!isnan, vals)
        if isempty(real_idx)
            per_bee_best_task[b] = -1
            per_bee_best_value[b] = NaN
            per_bee_second_value[b] = NaN
            per_bee_spec_score[b] = NaN
            continue
        end
        real_vals = vals[real_idx]
        sorted_inds = sortperm(real_vals; rev=true)
        best_rel = sorted_inds[1]
        best_idx = real_idx[best_rel]
        best_val = vals[best_idx]
        second_val = length(sorted_inds) >= 2 ? real_vals[sorted_inds[2]] : NaN
        per_bee_best_task[b] = task_ids[best_idx]
        per_bee_best_value[b] = float(best_val)
        per_bee_second_value[b] = isnan(second_val) ? NaN : float(second_val)
        others = [vals[i] for i in 1:length(vals) if i != best_idx && !isnan(vals[i])]
        if isempty(others) || isnan(best_val)
            per_bee_spec_score[b] = NaN
        else
            per_bee_spec_score[b] = (best_val - mean(others)) / (best_val + eps)
        end
    end

    per_task_top_bee = Dict{Int,Any}()
    per_task_top_value = Dict{Int,Float64}()
    per_task_second_value = Dict{Int,Float64}()
    per_task_dominance = Dict{Int,Float64}()
    for tid in task_ids
        pairs = [(b, last_known[b][tid]) for b in keys(last_known) if !isnan(last_known[b][tid])]
        if isempty(pairs)
            per_task_top_bee[tid] = nothing
            per_task_top_value[tid] = NaN
            per_task_second_value[tid] = NaN
            per_task_dominance[tid] = NaN
            continue
        end
        sorted_pairs = sort(pairs, by = x -> x[2], rev=true)
        per_task_top_bee[tid] = sorted_pairs[1][1]
        per_task_top_value[tid] = float(sorted_pairs[1][2])
        per_task_second_value[tid] = length(sorted_pairs) >= 2 ? float(sorted_pairs[2][2]) : NaN
        if isnan(per_task_top_value[tid])
            per_task_dominance[tid] = NaN
        else
            s = isnan(per_task_second_value[tid]) ? 0.0 : per_task_second_value[tid]
            per_task_dominance[tid] = (per_task_top_value[tid] - s) / (per_task_top_value[tid] + eps)
        end
    end

    unique_top_bees = unique(collect(values(per_task_top_bee)))
    unique_top_bees = filter(x -> x !== nothing, unique_top_bees)
    n_covered = length(unique_top_bees)
    coverage_fraction = length(task_ids) == 0 ? 0.0 : n_covered / length(task_ids)

    counts = Dict{Any,Int}()
    for b in unique_top_bees
        counts[b] = 0
    end
    for tid in task_ids
        b = per_task_top_bee[tid]
        if b === nothing
            continue
        end
        counts[b] = get(counts, b, 0) + 1
    end
    H = 0.0
    assignment_entropy = NaN
    assignment_entropy_totalbees = NaN
    if length(task_ids) > 0 && !isempty(counts)
        for (_, c) in counts
            p = c / length(task_ids)
            H -= p * (p > 0 ? Base.log(p) : 0.0)
        end
        # normalize by number of distinct top-bees (K) -- previous behaviour
        K = max(length(keys(counts)), 1)
        maxH = Base.log(K)
        assignment_entropy = maxH == 0 ? 0.0 : H / maxH
        # also provide entropy normalized by total number of bees (N)
        N = length(unique(vcat(log[!, bee1col], log[!, bee2col])))
        assignment_entropy_totalbees = N <= 1 ? 0.0 : H / Base.log(N)
    end

    spec_values = [v for v in values(per_bee_spec_score) if !isnan(v)]
    specialization_score_mean = isempty(spec_values) ? NaN : mean(spec_values)

    return (
        last_known = last_known,
        per_bee_best_task = per_bee_best_task,
        per_bee_best_value = per_bee_best_value,
        per_bee_second_value = per_bee_second_value,
        per_bee_spec_score = per_bee_spec_score,
        per_task_top_bee = per_task_top_bee,
        per_task_top_value = per_task_top_value,
        per_task_second_value = per_task_second_value,
        per_task_dominance = per_task_dominance,
        coverage_fraction = coverage_fraction,
        assignment_entropy = assignment_entropy,
        assignment_entropy_totalbees = assignment_entropy_totalbees,
        specialization_score_mean = specialization_score_mean
    )
end


"""
compute_run_quality(p_end, coverage, entropy_total; alpha=1.0, beta=1.0, p_min=0.0, p_max=1.0)

Compute a multiplicative run-quality score using:
 - p_end: final run performance (expected in [p_min,p_max], default [0,1])
 - coverage: coverage_fraction in [0,1]
 - entropy_total: assignment entropy normalized by total bees in [0,1]

Score = p_scaled * (coverage^alpha) * (entropy_total^beta)
where p_scaled = clamp((p_end - p_min)/(p_max - p_min), 0, 1).
Returns (score, components::Dict).
Assumes task values are in a scale where p_max is known (default 1.0). If your task values use a different range, pass p_min/p_max.
"""
function compute_run_quality(p_end::Real, coverage::Real, entropy_total::Real; alpha::Real=1.0, beta::Real=1.0, p_min::Real=0.0, p_max::Real=1.0)
    # scale p_end into [0,1]
    if p_max <= p_min
        error("compute_run_quality: invalid p_min/p_max")
    end
    p_scaled = clamp((p_end - p_min) / (p_max - p_min), 0.0, 1.0)
    cov = clamp(coverage, 0.0, 1.0)
    ent = clamp(entropy_total, 0.0, 1.0)
    score = p_scaled * (cov^alpha) * (ent^beta)
    components = Dict(
        :p_end => p_end,
        :p_scaled => p_scaled,
        :coverage => cov,
        :entropy_total => ent,
        :alpha => alpha,
        :beta => beta
    )
    return score, components
end


function run_summary(log::DataFrame; task_prefix::String="task_", timecol::Symbol=:time,
                     bee1col::Symbol=:bee1_id, bee2col::Symbol=:bee2_id,
                     taskidcol::Symbol=:task_id, atol=0.0)
    # Basic checks
    if nrow(log) == 0
        error("run_summary: empty log")
    end

    task_ids = extract_task_ids(log, task_prefix)
    if isempty(task_ids)
        error("run_summary: no task columns found with prefix=$(task_prefix)")
    end
    n_tasks = length(task_ids)

    # counts
    n_events = nrow(log)
    is_training = log[!, bee1col] .== log[!, bee2col]
    n_training = count(is_training)
    n_suppression = n_events - n_training

    # duration
    tmin = minimum(skipmissing(log[!, timecol])); tmax = maximum(skipmissing(log[!, timecol]))
    duration = float(tmax - tmin)

    # Best per-task across all bees/time
    per_task_best = compute_per_task_best(log, task_ids; task_prefix=task_prefix)

    # Primary run score: mean best across tasks
    run_score_mean_best_task = mean(collect(values(per_task_best)))

    per_bee_lastrow = compute_per_bee_lastrow(log; bee1col=bee1col, bee2col=bee2col, timecol=timecol)
    per_task_end_best = compute_per_task_end_best(per_bee_lastrow, task_ids; task_prefix=task_prefix)
    run_score_mean_best_task_at_end = mean(collect(values(per_task_end_best)))
    per_task_end_components = deepcopy(per_task_end_best)
    # compute specialization metrics via helper
    spec = compute_specialization_metrics(log, task_ids; task_prefix=task_prefix,
                                          bee1col=bee1col, bee2col=bee2col, timecol=timecol)

    

    # mean task value at interaction: use the task value for bee1 (the actor) at each event
    # We assume the event row contains the correct task column value for the bee being considered.
    mean_task_value_at_interaction, task_vals_at_event = compute_task_values_at_events(log; task_prefix=task_prefix, taskidcol=taskidcol)

    # mean gain per training: compute post - pre for the trained task by comparing
    # each training row's task value (post-event) with the previous row for the same
    # bee & task (pre-event). This is correct because task values are recorded after events.
    gains, mean_gain_per_training = compute_training_gains(log; task_prefix=task_prefix,
                                                           bee1col=bee1col, bee2col=bee2col,
                                                           taskidcol=taskidcol, timecol=timecol)

    # per-task mean gain (same technique per task)
    per_task_gains = compute_per_task_gains(log, task_ids; task_prefix=task_prefix,
                                            bee1col=bee1col, bee2col=bee2col, taskidcol=taskidcol, timecol=timecol)

    ## final mean accuracy: for each task take last observed value across rows and average across tasks
    #final_accs = Float64[]
    ## last observed per task = value at row with maximum time
    #for tid in task_ids
        #col = Symbol(task_prefix * string(tid))
        ## get rows where that column not missing
        #colvals = skipmissing(Float64.(log[!, col]))
        ## best approximation to end-of-run: take value from row with max time (in case some tasks not present at end)
        ## find index of last non-missing row for this column
        #idxs = findall(!ismissing.(log[!, col]))
        #if isempty(idxs)
            #push!(final_accs, NaN)
            #continue
        #end
        #lastidx = idxs[argmax(log[idxs, timecol])]
        #push!(final_accs, float(log[lastidx, col]))
    #end
    #final_mean_accuracy = mean(final_accs)

    # per-bee stats
    n_bees = length(unique(vcat(log[!, bee1col], log[!, bee2col])))

    # specialization metrics
    spec = compute_specialization_metrics(log, task_ids; task_prefix=task_prefix,
                                          bee1col=bee1col, bee2col=bee2col, timecol=timecol)

    # compute multiplicative run-quality score using final performance, coverage and total-bee entropy
    # assume task values are in [0,1] by default; adjust p_min/p_max if different
    p_end = run_score_mean_best_task_at_end
    coverage = spec.coverage_fraction
    entropy_total = spec.assignment_entropy_totalbees
    run_quality_score, run_quality_components = compute_run_quality(p_end, coverage, entropy_total)

    return (
        run_score_mean_best_task = run_score_mean_best_task,
        n_events = n_events,
        n_training = n_training,
        n_suppression = n_suppression,
        duration = duration,
        n_bees = n_bees,
        n_tasks = n_tasks,
        per_task_best = per_task_best,
        run_mean_task_value_at_interaction = mean_task_value_at_interaction,
        mean_gain_per_training = mean_gain_per_training,
        per_task_mean_gain = per_task_gains,
        #final_mean_accuracy = final_mean_accuracy,
        run_score_mean_best_task_at_end = run_score_mean_best_task_at_end,
        per_task_end_components = per_task_end_components
        ,
        # specialization / coverage outputs (from compute_specialization_metrics)
        per_bee_best_task = spec.per_bee_best_task,
        per_bee_spec_score = spec.per_bee_spec_score,
        per_task_top_bee = spec.per_task_top_bee,
        per_task_dominance = spec.per_task_dominance,
        coverage_fraction = spec.coverage_fraction,
        assignment_entropy = spec.assignment_entropy,
        assignment_entropy_totalbees = spec.assignment_entropy_totalbees,
        specialization_score_mean = spec.specialization_score_mean,
        last_known = spec.last_known
        ,
        run_quality_score = run_quality_score,
        run_quality_components = run_quality_components
    )
end

















