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
        seen = falses(n_tasks)  # BitVector ok
        if dfs(bee, seen)
            matched += 1
            matched == n_tasks && return true
        end
    end
    return matched == n_tasks
end

"""
Event-based onset:
Maintains last-known accuracies per bee and checks for a full distinct-bee-per-task matching.
"""
function analyze_event_log_df(df::DataFrame; thresholds = [0.8, 0.9])
    task_cols = filter(c -> startswith(String(c), "task_") && String(c) != "task_id", names(df))
    isempty(task_cols) && error("No task_* columns found (excluding task_id).")

    n_tasks = length(task_cols)
    t_end = maximum(df.time)

    # Ensure chronological order (important!)
    sort!(df, :time)

    # All bees that ever appear
    bees = sort(unique(df.bee1_id))
    n_bees_total = length(bees)

    # Map bee id -> row index in A
    bee_to_idx = Dict(b => i for (i, b) in enumerate(bees))

    # Store last-known accuracy vectors; start as missing
    last = Matrix{Union{Missing, Float64}}(missing, n_bees_total, n_tasks)


    onset = Dict{Float64, Union{Missing, Float64}}(thr => missing for thr in thresholds)
    remaining = Set(thresholds)

    # Track how many bees we’ve seen at least once (so last row not missing)
    seen_bee = falses(n_bees_total)
    n_seen = 0

    for r in eachrow(df)
        bi = bee_to_idx[r.bee1_id]

        # Update state for this bee
        vals = collect(Float64, r[task_cols])
        last[bi, :] = vals

        if !seen_bee[bi]
            seen_bee[bi] = true
            n_seen += 1
        end

        # Need at least n_tasks distinct bees initialized to even have a chance
        n_seen < n_tasks && continue

        # Build current A from initialized bees only (those not missing)
        active_idxs = findall(seen_bee)  # bees with known state
        A = Matrix{Float64}(last[active_idxs, :])

        # Evaluate thresholds still missing
        for thr in collect(remaining)
            if has_full_matching(A .>= thr)
                onset[thr] = r.time
                delete!(remaining, thr)
            end
        end

        isempty(remaining) && break
    end

    out = DataFrame(n_tasks = n_tasks, n_bees_total = n_bees_total, t_end = t_end)
    for thr in thresholds
        col = Symbol("onset_thr_", replace(string(thr), "." => "p"))
        out[!, col] = [onset[thr]]
    end

    return out
end

function analyze_event_log_csv(path::AbstractString; thresholds = [0.8, 0.9])
    df = CSV.read(path, DataFrame)
    return analyze_event_log_df(df; thresholds=thresholds)
end

"--------------------------------------------------"

using CSV, DataFrames, Statistics

# Shannon entropy of a probability vector p (sum(p)=1), ignoring zeros
function shannon_entropy(p::AbstractVector{<:Real})
    h = 0.0
    for x in p
        if x > 0
            h -= x * log(x)  # natural log; normalization cancels base anyway
        end
    end
    return h
end

# Return specialization score in [0,1]: 1 - H/Hmax
# v are nonnegative "weights" (accuracies); we normalize them to probabilities.
function specialization_score(v::AbstractVector{<:Real})
    s = sum(v)
    s == 0 && return missing
    p = v ./ s
    n = length(p)
    n <= 1 && return 1.0  # degenerate case
    h = shannon_entropy(p)
    hmax = log(n)
    return 1 - h / hmax
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

    # last-known accuracies per bee
    last = Matrix{Union{Missing, Float64}}(missing, n_bees_total, n_tasks)

    onset = Dict{Float64, Union{Missing, Float64}}(thr => missing for thr in thresholds)
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

        # onset check only when we have enough bees initialized
        n_seen < n_tasks && continue

        active_idxs = findall(seen_bee)
        A = Matrix{Float64}(last[active_idxs, :])  # safe: only initialized bees

        for thr in collect(remaining)
            if has_full_matching(A .>= thr)
                onset[thr] = r.time
                delete!(remaining, thr)
            end
        end

        isempty(remaining) && break
    end

    # Build FINAL state matrix for specialization (use all bees that have a known state)
    active_idxs = findall(seen_bee)
    Afinal = Matrix{Float64}(last[active_idxs, :])
    n_bees_final = size(Afinal, 1)

    # Task specialization: one score per task, then summary
    task_specs = [specialization_score(view(Afinal, :, k)) for k in 1:n_tasks]
    task_spec_mean = all(ismissing, task_specs) ? missing : mean(skipmissing(task_specs))
    task_spec_min  = all(ismissing, task_specs) ? missing : minimum(skipmissing(task_specs))
    task_spec_max  = all(ismissing, task_specs) ? missing : maximum(skipmissing(task_specs))

    # Agent specialization: one score per bee, then summary
    bee_specs = [specialization_score(view(Afinal, i, :)) for i in 1:n_bees_final]
    bee_spec_mean = all(ismissing, bee_specs) ? missing : mean(skipmissing(bee_specs))
    bee_spec_min  = all(ismissing, bee_specs) ? missing : minimum(skipmissing(bee_specs))
    bee_spec_max  = all(ismissing, bee_specs) ? missing : maximum(skipmissing(bee_specs))

    out = DataFrame(
        n_tasks = n_tasks,
        n_bees_total = n_bees_total,
        n_bees_final = n_bees_final,
        t_end = t_end,
        task_spec_mean = task_spec_mean,
        task_spec_min = task_spec_min,
        task_spec_max = task_spec_max,
        agent_spec_mean = bee_spec_mean,
        agent_spec_min = bee_spec_min,
        agent_spec_max = bee_spec_max,
    )

    for thr in thresholds
        col = Symbol("onset_thr_", replace(string(thr), "." => "p"))
        out[!, col] = [onset[thr]]
    end

    return out
end

function analyze_event_log_csv(path::AbstractString; thresholds = [0.8, 0.9])
    df = CSV.read(path, DataFrame)
    return analyze_event_log_df(df; thresholds=thresholds)
end
