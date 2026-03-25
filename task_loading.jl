using Random

task_symbol_to_id(s::AbstractString) = parse(Int, match(r"task_(\d+)$", s).captures[1])

function dataset_names_to_task_ids(names)::Vector{Int}
    return [task_symbol_to_id(String(n)) for n in names]
end

function find_parent_config_json(path::String; filename::String="config.json", max_up::Int=6)
    dir = abspath(path)
    isdir(dir) || (dir = dirname(dir))
    for _ in 1:max_up
        candidate = joinpath(dir, filename)
        if isfile(candidate)
            return candidate
        end
        dir = dirname(dir)
    end
    error("Could not find $filename by walking up from $path (max_up=$max_up)")
end

"""
Select task indices from 1:n_total.

mode = :first  -> 1:n_tasks
mode = :random -> random subset of size n_tasks from 1:n_total (sorted)
"""
function select_task_indices(n_tasks::Int, n_total::Int;
                            mode::Symbol = :first,
                            rng::AbstractRNG = Random.default_rng())
    @assert 1 ≤ n_tasks ≤ n_total
    if mode == :first
        return collect(1:n_tasks)
    elseif mode == :random
        return sort!(randperm(rng, n_total)[1:n_tasks])
    else
        error("Unknown task selection mode: $mode")
    end
end

"""
Given an initial set of selected indices, choose ONE new task to add.

strategy = :next -> the smallest index not already in selected (often n_tasks+1 if :first)
strategy = :random_remaining -> uniform random from remaining pool
"""
function select_added_task(selected::Vector{Int}, n_total::Int;
                          strategy::Symbol = :next,
                          rng::AbstractRNG = Random.default_rng())
    chosen = Set(selected)
    remaining = [i for i in 1:n_total if !(i in chosen)]
    isempty(remaining) && error("No remaining tasks to add (selected already covers all tasks).")

    if strategy == :next
        return first(remaining)  # because remaining is in increasing order
    elseif strategy == :random_remaining
        return remaining[rand(rng, 1:length(remaining))]
    else
        error("Unknown add-task strategy: $strategy")
    end
end

function choose_tasks_with_restart_support!(config::Dict, n_total::Int; rng::AbstractRNG)
    new_n = config["n_tasks"]

    if get(config, "load_models_from", "") != ""
        model_dir = config["load_models_from"]
        cfg_path = find_parent_config_json(model_dir)
        old_cfg = JSON3.read(read(cfg_path, String))

        old_task_ids = dataset_names_to_task_ids(old_cfg["dataset_names"])
        old_n = length(old_task_ids)

        if new_n < old_n
            # You *can* allow this, but it’s usually conceptually messy for stitching.
            # Choose your preferred behavior: error or truncate.
            error("New run requests n_tasks=$new_n but previous run had $old_n tasks. Refusing to drop tasks for a restart.")
            # If you prefer truncation instead, do:
            # old_task_ids = old_task_ids[1:new_n]
        end

        # If we need more tasks, sample from remaining
        chosen = copy(old_task_ids)
        added = Int[]

        if new_n > old_n
            remaining = setdiff(collect(1:n_total), chosen)
            k = new_n - old_n
            k ≤ length(remaining) || error("Not enough remaining tasks to add: need $k, have $(length(remaining)).")

            # Strategy: random from remaining (you can also support :next if you want)
            perm = randperm(rng, length(remaining))
            added = remaining[perm[1:k]]
            append!(chosen, added)
        end

        config["task_ids"] = chosen
        config["dataset_names"] = Symbol.("task_$(i)" for i in chosen)

        # for stitching / provenance
        config["restart_base_task_ids"] = old_task_ids
        config["restart_added_task_ids"] = added
        config["restart_source_config"] = cfg_path

        println("Restart detected. Base tasks (old): ", old_task_ids)
        println("Added tasks (new): ", added)
        println("Final task_ids: ", chosen)
        return chosen
    else
        # fresh run: use your existing selection logic (first/random)
        mode = get(config, "task_selection_mode", "first") |> Symbol
        task_ids = select_task_indices(new_n, n_total; mode=mode, rng=rng)
        config["task_ids"] = task_ids
        config["dataset_names"] = Symbol.("task_$(i)" for i in task_ids)
        println("Fresh run. task_ids: ", task_ids)
        return task_ids
    end
end
