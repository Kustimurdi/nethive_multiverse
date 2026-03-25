using CSV, DataFrames, Statistics

"Per-task best accuracies and their min/mean across tasks."
function task_best_stats(A::AbstractMatrix{<:Real})
    best_per_task = vec(maximum(A, dims=1))  # length = n_tasks
    return (
        final_min_task_best = minimum(best_per_task),
        final_mean_task_best = mean(best_per_task),
    )
end

"""
Maximum threshold thr for which a full matching exists in A (final state).
Searches over unique accuracies in A for an exact answer.
"""
function max_thr_matching(A::AbstractMatrix{<:Real})
    vals = sort!(unique(vec(Float64.(A))))
    isempty(vals) && return missing

    lo, hi = 1, length(vals)
    best = vals[1]

    while lo <= hi
        mid = (lo + hi) ÷ 2
        thr = vals[mid]
        if has_full_matching(A .>= thr)
            best = thr
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    return best
end


function shannon_entropy(p::AbstractVector{<:Real})
    h = 0.0
    for x in p
        if x > 0
            h -= x * log(x)
        end
    end
    return h
end

function specialization_score(v::AbstractVector{<:Real}; eps::Float64 = 1e-6)
    w = max.(Float64.(v), 0.0) .+ eps
    p = w ./ sum(w)
    n = length(p)
    n <= 1 && return 1.0
    h = shannon_entropy(p)
    return 1 - h / log(n)
end


"Compute mean task- and agent-specialization scores for an accuracy matrix A (n_bees × n_tasks)."
function specialization_means(A::AbstractMatrix{<:Real})
    n_bees, n_tasks = size(A)
    task_specs = [specialization_score(view(A, :, k)) for k in 1:n_tasks]
    bee_specs  = [specialization_score(view(A, i, :)) for i in 1:n_bees]

    task_mean = all(ismissing, task_specs) ? missing : mean(skipmissing(task_specs))
    bee_mean  = all(ismissing, bee_specs)  ? missing : mean(skipmissing(bee_specs))
    return (task_mean = task_mean, agent_mean = bee_mean)
end

function has_full_matching(adj::AbstractMatrix{Bool})
    n_bees, n_tasks = size(adj)
    n_bees < n_tasks && return false

    match_task = zeros(Int, n_tasks)

    function dfs(bee::Int, seen::AbstractVector{Bool})
        for task in 1:n_tasks
            if adj[bee, task] && !seen[task]
                seen[task] = true
                if match_task[task] == 0 || dfs(match_task[task], seen)
                    match_task[task] = bee
                    return true
                end
            end
        end
        return false
    end

    matched = 0
    for bee in 1:n_bees
        seen = falses(n_tasks)
        if dfs(bee, seen)
            matched += 1
            matched == n_tasks && return true
        end
    end
    return matched == n_tasks
end

function analyze_event_log_df(df::DataFrame; thresholds = [0.8, 0.9])
    task_cols = filter(c -> startswith(String(c), "task_") && String(c) != "task_id", names(df))
    isempty(task_cols) && error("No task_* columns found (excluding task_id).")

    n_tasks = length(task_cols)
    t_end = maximum(df.time)

    sort!(df, :time)

    bees = sort(unique(df.bee1_id))
    n_bees_total = length(bees)
    bee_to_idx = Dict(b => i for (i, b) in enumerate(bees))

    last = Matrix{Union{Missing, Float64}}(missing, n_bees_total, n_tasks)

    # onset times + specialization-at-onset snapshots
    onset_time = Dict{Float64, Union{Missing, Float64}}(thr => missing for thr in thresholds)
    onset_task_spec_mean = Dict{Float64, Union{Missing, Float64}}(thr => missing for thr in thresholds)
    onset_agent_spec_mean = Dict{Float64, Union{Missing, Float64}}(thr => missing for thr in thresholds)

    remaining = Set(thresholds)

    seen_bee = falses(n_bees_total)
    n_seen = 0

    for r in eachrow(df)
        bi = bee_to_idx[r.bee1_id]
        vals = collect(Float64, r[task_cols])
        last[bi, :] = vals

        if !seen_bee[bi]
            seen_bee[bi] = true
            n_seen += 1
        end

        n_seen < n_tasks && continue

        active_idxs = findall(seen_bee)
        A = Matrix{Float64}(last[active_idxs, :])

        # check thresholds still missing
        for thr in collect(remaining)
            if has_full_matching(A .>= thr)
                onset_time[thr] = r.time

                specs = specialization_means(A)
                onset_task_spec_mean[thr] = specs.task_mean
                onset_agent_spec_mean[thr] = specs.agent_mean

                delete!(remaining, thr)
            end
        end

        isempty(remaining) && break
    end

    # final specialization (using last-known state at end)
    active_idxs = findall(seen_bee)
    Afinal = Matrix{Float64}(last[active_idxs, :])
    final_specs = specialization_means(Afinal)
    tb = task_best_stats(Afinal)
    thrmax = max_thr_matching(Afinal)

    out = DataFrame(
        n_tasks = n_tasks,
        n_bees_total = n_bees_total,
        n_bees_final = size(Afinal, 1),
        t_end = t_end,
        task_spec_mean_final = final_specs.task_mean,
        agent_spec_mean_final = final_specs.agent_mean,
        final_min_task_best = tb.final_min_task_best,
        final_mean_task_best = tb.final_mean_task_best,
        max_thr_matching_final = thrmax,
    )

    for thr in thresholds
        thr_name = replace(string(thr), "." => "p")
        out[!, Symbol("onset_thr_", thr_name)] = [onset_time[thr]]
        out[!, Symbol("task_spec_mean_onset_thr_", thr_name)] = [onset_task_spec_mean[thr]]
        out[!, Symbol("agent_spec_mean_onset_thr_", thr_name)] = [onset_agent_spec_mean[thr]]
    end

    return out
end

function analyze_event_log_csv(path::AbstractString; thresholds = [0.8, 0.9])
    df = CSV.read(path, DataFrame)
    return analyze_event_log_df(df; thresholds=thresholds)
end
