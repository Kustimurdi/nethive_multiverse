include("./steffen_score.jl")

function steffen_scores_all_runs_from_config_long(roots::Vector{String};
    thresholds = [0.8, 0.9],
    time_points = nothing,                 # NEW
    event_filename = "states.csv",        # UPDATED
    config_filename = "config.json",
)
    runs = collect_runs_long(roots; event_filename=event_filename, config_filename=config_filename)
    n = length(runs)

    results_vec = Vector{Union{Nothing,DataFrame}}(fill(nothing, n))
    failures = Vector{NamedTuple}(undef, 0)
    fail_lock = ReentrantLock()

    p = Progress(n; desc = "Analyzing runs")

    Threads.@threads for i in 1:n
        run = runs[i]
        try
            results_vec[i] = steffen_scores_one_run_long(run;
                thresholds = thresholds,
                time_points = time_points,
                hold_epochs = 0
            )
        catch e
            lock(fail_lock) do
                push!(failures, (
                    root = run.root,
                    param_folder = run.param_folder,
                    param_dir = run.param_dir,
                    #config_path = run.config_path,
                    error = sprint(showerror, e),
                    # optional: include backtrace for debugging
                    # backtrace = sprint(Base.show_backtrace, catch_backtrace()),
                ))
            end
        end

        next!(p)

    end

    rows = [df for df in results_vec if df !== nothing]
    results = isempty(rows) ? DataFrame() : vcat(rows...; cols=:union)
    failures_df = isempty(failures) ? DataFrame() : DataFrame(failures)

    return results, failures_df
end

function collect_runs_long(roots::Vector{String};
    event_filename::AbstractString = "states.csv",
    config_filename::AbstractString = "config.json",
)
    runs = NamedTuple[]
    for root in roots
        for param_dir in find_param_dirs(root)
            param_folder = splitdir(param_dir)[2]
            event_path  = joinpath(param_dir, event_filename)
            config_path = joinpath(param_dir, config_filename)
            push!(runs, (
                root = root,
                param_dir = param_dir,
                param_folder = param_folder,
                data_path = event_path,
                config_path = config_path,
            ))
        end
    end
    return runs
end

function steffen_scores_one_run_long(run; thresholds=[0.8, 0.9],
                                time_points=nothing, hold_epochs=0)
    # sanity checks
    isfile(run.config_path)    || error("missing config.json")
    isfile(run.data_path)      || error("missing data path, prob. states.csv")

    # --- load only selected config values (as a 1-row DF) ---
    cfg_all = config_to_df(run.config_path)  # 1-row DF
    keep_cfg = ["n_epochs", "learning_rate", "interaction_rate", "dead_time", "lambda_sensitivity", "n_bees", "n_tasks"]

    # keep only columns that actually exist (avoids crashes if one key missing)
    cfg = select(cfg_all, intersect(keep_cfg, names(cfg_all)))

    # --- load event log (DataFrame) ---
    acc_log = CSV.read(run.data_path, DataFrame)  # adjust if you already have a loader


    # --- NEW: time-to-specialization for each metric/threshold ---
    times = specialization_times_df_long(acc_log;
        thresholds = thresholds,
        hold_epochs = hold_epochs,
    )

    # pack into a 1-row DF with explicit column names, e.g. t_any_geq_q0_8
    time_cols = Dict{Symbol,Any}()
    for r in eachrow(times)
        qstr = replace(string(r.threshold), "." => "_")
        col = Symbol("t_", r.metric, "_q", qstr)
        time_cols[col] = r.t_reach
    end
    times_row = DataFrame(time_cols)


    if time_points === nothing
        #tmax = parse(Float64, cfg_all[1, :n_epochs])  # assuming n_epochs corresponds to max time
        tmax = Float64(cfg_all[1, :n_epochs])
        time_points = [tmax / 2, tmax]
    end

    # --- compute analysis (many rows) ---
    ana = steffen_scores_over_time_df_long(acc_log;
        time_points=time_points,
        thresholds=thresholds
    )
    n = nrow(ana)

    # --- repeat cfg + metadata to match ana rows ---
    # repeat the 1-row cfg to n rows
    cfg_rep = repeat(cfg, n)

    # metadata columns (length n)
    meta = DataFrame(
        root          = fill(run.root, n),
        param_folder  = fill(run.param_folder, n),
        param_dir     = fill(run.param_dir, n),
        #config_path    = fill(run.config_path, n),
    )

    times_rep = repeat(times_row, n)

    @assert cfg_rep isa DataFrame
    @assert times_rep isa DataFrame
    @assert ana isa DataFrame
    @assert meta isa DataFrame

    # final merged table
    out = hcat(meta, cfg_rep, times_rep, ana; makeunique=true)

    return out
end

function steffen_scores_over_time_df_long(acc_log::DataFrame;
    time_points::AbstractVector{<:Real},
    agent_col::Symbol = :bee_id,
    time_col::Symbol  = :epoch,
    thresholds = [0.8, 0.9]
)
    out = DataFrame(
        time = Float64[],
        threshold = Float64[],
        any_geq = Int[],
        exactly_one_geq = Int[],
        unique_specialist = Int[],
        avg_agents_geq_per_task = Float64[],
    )

    for t in time_points
        state = state_at_time_long(acc_log, t;
            agent_col = agent_col,
            time_col  = time_col,
            acc_col   = :accuracies
        )

        for q in thresholds
            stats = task_coverage_stats_long(state, q)
            push!(out, (
                Float64(t),
                Float64(q),
                stats.any_geq,
                stats.exactly_one_geq,
                stats.unique_specialist,
                stats.avg_agents_geq_per_task
            ))
        end
    end

    return out
end

function state_at_time_long(acc_log::DataFrame, t::Real;
    agent_col::Symbol = :bee_id,
    task_col::Symbol  = :task_id,
    time_col::Symbol  = :epoch,        # or :time, depending on your df
    acc_col::Symbol   = :accuracies
)
    # keep only rows up to time t
    sub = acc_log[acc_log[!, time_col] .<= t, :]
    isempty(sub) && return DataFrame()

    # last-known row per (agent, task) within sub
    last_rows = unique(
        sort(sub, [agent_col, task_col, time_col], rev = [false, false, true]),
        [agent_col, task_col]
    )

    # keep only relevant columns (and sort nicely)
    select!(last_rows, [time_col, agent_col, task_col, acc_col])
    sort!(last_rows, [agent_col, task_col])

    return last_rows
end

function task_matrix_long(state::DataFrame;
    agent_col::Symbol = :bee_id,
    task_col::Symbol  = :task_id,
    acc_col::Symbol   = :accuracies,
)
    agents = sort(unique(state[!, agent_col]))
    tasks  = sort(unique(state[!, task_col]))

    xi = Dict(a => i for (i, a) in enumerate(agents))
    yi = Dict(t => j for (j, t) in enumerate(tasks))

    A = fill(NaN, length(agents), length(tasks))

    for r in eachrow(state)
        A[xi[r[agent_col]], yi[r[task_col]]] = r[acc_col]
    end

    return A, agents, tasks
end

function n_tasks_with_any_geq_long(state::DataFrame, q::Real)
    A, _ = task_matrix_long(state)
    # for each task (column), check if any agent meets threshold
    count_task = count(j -> any(x -> isfinite(x) && x ≥ q, view(A, :, j)), 1:size(A,2))
    return count_task#, count_task/size(A,2)
end

function avg_agents_geq_per_task_long(state::DataFrame, q::Real)
    A, _ = task_matrix_long(state)

    counts_per_task = [
        count(x -> isfinite(x) && x >= q, view(A, :, j))
        for j in 1:size(A, 2)
    ]

    return mean(counts_per_task)
end

function n_tasks_with_exactly_one_geq_long(state::DataFrame, q::Real)
    A, _ = task_matrix_long(state)
    count_task = count(j -> begin
        n = count(x -> isfinite(x) && x ≥ q, view(A, :, j))
        n == 1
    end, 1:size(A,2))
    return count_task#, count_task/size(A,2)
end


function n_tasks_with_unique_specialist_long(state::DataFrame, q::Real)
    A, _ = task_matrix_long(state)
    n_agents, n_tasks = size(A)

    count_task = 0
    for j in 1:n_tasks
        # which agents meet threshold on task j?
        good = [i for i in 1:n_agents if isfinite(A[i,j]) && A[i,j] ≥ q]
        length(good) == 1 || continue
        i = only(good)

        v = A[i,j]
        # specialist condition: all other task scores strictly lower than v
        is_specialist = all(k -> (k == j) || (!isfinite(A[i,k]) ? true : A[i,k] < v), 1:n_tasks)

        if is_specialist
            count_task += 1
        end
    end
    return count_task#, count_task/n_tasks
end

function task_coverage_stats_long(state::DataFrame, q::Real)
    return (
        any_geq = n_tasks_with_any_geq_long(state, q),
        exactly_one_geq = n_tasks_with_exactly_one_geq_long(state, q),
        unique_specialist = n_tasks_with_unique_specialist_long(state, q),
        avg_agents_geq_per_task = avg_agents_geq_per_task_long(state, q)
    )
end



# time anaysis 

using DataFrames, Statistics

"""
Compute metrics from the current state dict `last_acc[(bee,task)] = acc`.

Returns NamedTuple with the same fields you use:
(any_geq, exactly_one_geq, unique_specialist, avg_agents_geq_per_task)
"""
function coverage_stats_from_state!(
    last_acc::Dict{Tuple{Int,Int}, Float64},
    bees::Vector{Int},
    tasks::Vector{Int},
    q::Real,
)
    n_bees  = length(bees)
    n_tasks = length(tasks)

    # Build a dense matrix A[bee_idx, task_idx] with NaN for missing
    A = fill(NaN, n_bees, n_tasks)
    bee_index  = Dict(b => i for (i,b) in enumerate(bees))
    task_index = Dict(t => j for (j,t) in enumerate(tasks))

    @inbounds for ((b,t), acc) in last_acc
        i = get(bee_index, b, 0)
        j = get(task_index, t, 0)
        if i != 0 && j != 0
            A[i,j] = acc
        end
    end

    # any_geq
    any_geq = 0
    # exactly_one_geq
    exactly_one_geq = 0
    # avg_agents_geq_per_task
    counts_per_task = zeros(Int, n_tasks)

    @inbounds for j in 1:n_tasks
        c = 0
        mx = -Inf
        for i in 1:n_bees
            x = A[i,j]
            if isfinite(x)
                mx = max(mx, x)
                if x >= q
                    c += 1
                end
            end
        end
        if mx >= q
            any_geq += 1
        end
        if c == 1
            exactly_one_geq += 1
        end
        counts_per_task[j] = c
    end

    avg_agents_geq_per_task = mean(counts_per_task)

    # unique_specialist (your definition)
    unique_specialist = 0
    @inbounds for j in 1:n_tasks
        good = Int[]
        for i in 1:n_bees
            x = A[i,j]
            if isfinite(x) && x >= q
                push!(good, i)
            end
        end
        length(good) == 1 || continue
        i = only(good)
        v = A[i,j]

        is_spec = true
        for k in 1:n_tasks
            k == j && continue
            x = A[i,k]
            if isfinite(x) && !(x < v)
                is_spec = false
                break
            end
        end
        if is_spec
            unique_specialist += 1
        end
    end

    return (
        any_geq = any_geq,
        exactly_one_geq = exactly_one_geq,
        unique_specialist = unique_specialist,
        avg_agents_geq_per_task = avg_agents_geq_per_task,
    )
end

"""
First epoch when metric reaches target (typically all tasks) for a given threshold q.

- metric ∈ (:any_geq, :exactly_one_geq, :unique_specialist)
- hold_epochs: require the condition to hold for this many consecutive epochs (robustness)
Returns `missing` if never reached.
"""
function time_to_specialization_long(
    acc_log::DataFrame;
    q::Real,
    metric::Symbol,
    hold_epochs::Int = 0,
    agent_col::Symbol = :bee_id,
    task_col::Symbol  = :task_id,
    time_col::Symbol  = :epoch,
    acc_col::Symbol   = :accuracies,
)
    metric ∈ (:any_geq, :exactly_one_geq, :unique_specialist) ||
        error("metric must be :any_geq, :exactly_one_geq, or :unique_specialist")

    # tasks/bees present in this run (this is what "all tasks" should mean)
    bees  = sort!(unique(acc_log[!, agent_col]))
    tasks = sort!(unique(acc_log[!, task_col]))
    target = length(tasks)

    # Sort by time ascending so we can update incrementally
    df = sort(acc_log, time_col)

    # last known accuracy for each (bee,task)
    last_acc = Dict{Tuple{Int,Int}, Float64}()

    # Walk epoch by epoch (only epochs that appear)
    epochs = sort!(unique(df[!, time_col]))

    # To process rows epoch-wise efficiently:
    g = groupby(df, time_col)

    streak = 0
    for t in epochs
        rows = g[(t,)]  # SubDataFrame for this epoch

        # update state with all observations at this epoch
        for r in eachrow(rows)
            last_acc[(Int(r[agent_col]), Int(r[task_col]))] = Float64(r[acc_col])
        end

        stats = coverage_stats_from_state!(last_acc, bees, tasks, q)
        value = getfield(stats, metric)

        if value == target
            streak += 1
            if streak > hold_epochs
                return t
            end
        else
            streak = 0
        end
    end

    return missing
end

"""
Compute times for all thresholds + metrics and return a tidy DataFrame.
"""
function specialization_times_df_long(acc_log::DataFrame;
    thresholds = [0.8, 0.9],
    metrics = [:any_geq, :exactly_one_geq, :unique_specialist],
    hold_epochs::Int = 0,
)
    out = DataFrame(threshold=Float64[], metric=Symbol[], t_reach=Union{Missing,Int}[])
    for q in thresholds
        for m in metrics
            t = time_to_specialization_long(acc_log; q=q, metric=m, hold_epochs=hold_epochs)
            push!(out, (Float64(q), m, t))
        end
    end
    return out
end