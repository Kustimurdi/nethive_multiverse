#using Pkg
#Pkg.activate("./env_nethive_multiverse/")
#Pkg.instantiate()

# Load packages directly (they should be available via JULIA_PROJECT)
@info "DEPOT_PATH" DEPOT_PATH

using ArgParse
using CSV
using DataFrames
using Distributions
using Flux
using JSON3
using LinearAlgebra
using MLDatasets
using Random
using Statistics
using Dates
using JLD2

# Load our modules
include("src/data/loaders.jl")
include("src/data/synthetic.jl")
include("src/core/definitions.jl")
include("src/core/multitask_training.jl")
include("src/core/methods.jl")
include("src/core/save_data.jl")
include("src/data/prepare_gaussset.jl")
include("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/task_loading.jl")

function memtag(tag)
    println("\n=== $tag ===")
    GC.gc()
    println("time: ", time())
    println("Sys.total_memory(): ", Sys.total_memory() ÷ 2^30, " GiB")
    println("Sys.free_memory():  ", Sys.free_memory()  ÷ 2^30, " GiB")
    println("GC live bytes:      ", Base.gc_live_bytes() ÷ 2^20, " MiB")
    println("maxrss (kB):        ", Sys.maxrss())
end

function parse_commandline()
    s = ArgParseSettings(description = "Run Gillespie NN simulations with flexible configuration")
    
    @add_arg_table! s begin
        "--config", "-c"
            help = "JSON configuration file path"
            arg_type = String
            default = ""
        "--output-dir", "-o"
            help = "Output directory for results"
            arg_type = String
            default = "results/"
        "--base-name", "-n"
            help = "Base name for output folder"
            arg_type = String
            default = "multiverse"
        "--timestamp"
            help = "Add timestamp to output files"
            action = :store_true
        "--seed", "-s"
            help = "Random seed for reproducibility (auto-generated if not specified)"
            arg_type = Int
        "--verbose", "-v"
            help = "Verbose output"
            action = :store_true
        "--save-results"
            help = "Whether to save results (states and summary) to files"
            action = :store_true
        "--save-all"
            help = "Whether to save all results to files"
            action = :store_true
        # Direct parameter specification (overrides config file)
        "--n-bees"
            help = "Number of bees"
            arg_type = Int
        "--dataset-names"
            help = "Number of tasks"
            arg_type = String
            nargs = '*'
        "--n-epochs"
            help = "Number of training epochs"
            arg_type = Int
        "--n-steps"
            help = "Number of training steps per epoch"
            arg_type = Int
        "--production-rate"
            help = "Production rate"
            arg_type = Float64
        "--interaction-rate"
            help = "Interaction rate"
            arg_type = Float64
        "--lambda-sensitivity"
            help = "Lambda sensitivity parameter"
            arg_type = Float64
        "--punish-rate"
            help = "Punish rate"
            arg_type = Float64
        "--punishment"
            help = "Punishment type: resetting, time_out, none, gradient_ascend"
            arg_type = String
        "--dead-time"
            help = "Dead time for bees"
            arg_type = Float64
        # Initial conditions
        "--random-init"
            help = "Use random initial conditions"
            arg_type = Bool
            nargs = '?'
        "--batch-size"
            help = "Batch size for data loaders"
            arg_type = Int
            default = 32
        "--batches-per-step"
            help = "Number of batches per training step (if applicable)"
            arg_type = Int
            default = nothing
    end
    
    return parse_args(s)
end

function load_config_from_json(config_path::String)
    """Load simulation configuration from JSON file"""
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    
    config_data = JSON3.read(read(config_path, String))
    # Convert JSON3.Object to Dict with string keys
    return Dict(String(k) => v for (k, v) in config_data)
end

function create_default_config()
    """Create default configuration dictionary"""
    return Dict(
        "n_bees" => 5,
        "dataset_names" => [:mnist, :fashion_mnist, :cifar10],
        "n_epochs" => 10,
        "n_steps_per_epoch" => 5,
        "production_rate" => 1,
        "interaction_rate" => 20,
        "learning_rate" => 0.01,
        "punish_rate" => 0.1,
        "punishment" => "gradient_ascend",
        "lambda_sensitivity" => 100.0,
        "batch_size" => 32,
        "save_nn_epochs" => 0,
        "seed" => nothing,  # Will be auto-generated if not specified
        "batches_per_step" => nothing,
        "dead_time" => 1.0
    )
end

function merge_config_with_args(config::Dict, args::Dict)
    """Merge configuration with command line arguments (args override config)"""
    merged = copy(config)
    
    # Map command line arguments to config keys
    arg_mapping = Dict(
        "n-bees" => "n_bees",
        "n-tasks" => "n_tasks",
        "n-epochs" => "n_epochs",
        "n-steps" => "n_steps_per_epoch",
        "seed" => "seed",
        "production-rate" => "production_rate",
        "interaction-rate" => "interaction_rate",
        "lambda-sensitivity" => "lambda_sensitivity",
        "punishment" => "punishment",
        "batch-size" => "batch_size",
        "batches-per-step" => "batches_per_step",
        "dead-time" => "dead_time"
    )
    
    for (arg_key, config_key) in arg_mapping
        if haskey(args, arg_key) && (args[arg_key] !== nothing)
            merged[config_key] = args[arg_key]
        end
    end
    
    return merged
end

function initialize_hive_from_config(config::Dict, model_template::Function)
    """Initialize hive from configuration dictionary"""
    hive_config = MultiTaskHiveConfig(
        Symbol.(config["dataset_names"]),  # Convert to Vector{Symbol}
        model_template,
        config["max_input_dim"],
        config["max_output_dim"],
        config["n_bees"],
        config["n_epochs"],
        config["n_steps_per_epoch"],
        Float64(config["production_rate"]),
        Float64(config["interaction_rate"]),
        Float64(config["learning_rate"]),
        Float64(config["punish_rate"]),
        Float64(config["lambda_sensitivity"]),
        Symbol(config["punishment"]),
        config["seed"],
        config["save_nn_epochs"],
        config["batches_per_step"],
        Float64(config["dead_time"])
    )
    
    hive = MultiTaskHive(hive_config)

    return hive
end

function verify_loaded_model_once(model_template, loaded_model, model_state; features_dimension::Int)
    rng = MersenneTwister(0)
    x = rand(rng, Float32, features_dimension)   # shape matches Dense(in, out)

    # reference model built from template + same state
    ref = model_template()
    Flux.loadmodel!(ref, model_state)

    y_loaded = loaded_model(x)
    y_ref    = ref(x)

    println("Verifying loaded model matches saved state on a test input...")
    @assert y_loaded == y_ref  # should be EXACT for this architecture
    return true
end

function model_fingerprint(m; k=16)
    ps = collect(Flux.params(m))
    a = vec(Array(first(ps)))
    k = min(k, length(a))
    return a[1:k]
end

"""
    load_models_into_hive!(hive, model_dir; rng=Random.default_rng(), shuffle_if_subset=true)

Load pre-trained models into the hive from `model_dir`.

If the hive has fewer bees than the number of available saved models, a random subset
of saved bees is selected (unless `shuffle_if_subset=false`).

Returns a named tuple with metadata, in particular:
- `hive_to_source_bee` : Vector{Int}  (index = hive bee id, value = source bee id from model_dir)
- `source_to_hive_bee` : Dict{Int,Int}
- `available_source_bees` : Vector{Int}
"""
function load_models_into_hive!(
    hive::MultiTaskHive,
    model_dir::String;
    rng::AbstractRNG = Random.default_rng(),
    shuffle_if_subset::Bool = true,
)
    n_hive = length(hive.brains)

    # --- Find available saved bee models in directory ---
    # Expected pattern: bee_<id>_model.jdl2
    model_files = filter(f -> occursin(r"^bee_\d+_model\.jdl2$", f), readdir(model_dir))
    available_source_bees = collect(parse(Int, match(r"^bee_(\d+)_model\.jdl2$", f).captures[1]) for f in model_files)
    sort!(available_source_bees)

    n_available = length(available_source_bees)
    n_available == 0 && error("No model files found in $model_dir matching bee_<id>_model.jdl2")

    if n_hive > n_available
        error("Hive expects $n_hive bees, but only $n_available saved models were found in $model_dir")
    end

    # --- Choose which source bees to load ---
    selected_source_bees =
        if n_hive < n_available && shuffle_if_subset
            sort(randperm(rng, n_available)[1:n_hive] .|> i -> available_source_bees[i])
        else
            available_source_bees[1:n_hive]
        end

    @info "Loading $n_hive bee models into hive from $model_dir"
    @info "Available source bees: $(available_source_bees)"
    @info "Selected source bees:  $(selected_source_bees)"

    # Mapping: hive bee 1..n_hive -> source bee id in old run
    hive_to_source_bee = collect(selected_source_bees)
    source_to_hive_bee = Dict(src => hive_idx for (hive_idx, src) in enumerate(hive_to_source_bee))

    # --- Load selected models into hive slots 1:n_hive ---
    for (hive_idx, source_bee_id) in enumerate(hive_to_source_bee)
        model_path = joinpath(model_dir, "bee_$(source_bee_id)_model.jdl2")
        if isfile(model_path)
            @info "Loading source bee $(source_bee_id) into hive bee $(hive_idx) from $model_path"
            JLD2.@load model_path model_state
            before = model_fingerprint(hive.brains[hive_idx])
            Flux.loadmodel!(hive.brains[hive_idx], model_state)
            after  = model_fingerprint(hive.brains[hive_idx])
            @assert before != after 
            # Verify loaded model matches the saved state (sanity check)
            verify_loaded_model_once(hive.config.model_template, hive.brains[hive_idx], model_state; features_dimension=hive.config.max_input_dim)
        else
            error("Selected model file not found: $model_path")
        end
    end

    # --- Load suppression state and map rows consistently ---
    sup_path = joinpath(model_dir, "suppressed_tasks.jdl2")
    if isfile(sup_path)
        JLD2.@load sup_path suppressed_tasks suppression_time_left

        # Copy selected source rows into hive rows, and overlap in task dimension
        overwrite_overlap_rows_cols!(
            hive.suppressed_tasks,
            suppressed_tasks,
            hive_to_source_bee,        # source rows to take, one per hive row
        )

        overwrite_overlap_rows_cols!(
            hive.suppression_start_times,
            suppression_time_left,
            hive_to_source_bee,
        )

        println("suppression state loaded")
        println("quick test:")
        println("hive row 1 corresponds to source bee $(hive_to_source_bee[1])")
        println("suppressed equal? ", hive.suppressed_tasks[1,1] == suppressed_tasks[hive_to_source_bee[1],1])
        println("suppression time equal? ", hive.suppression_start_times[1,1] == suppression_time_left[hive_to_source_bee[1],1])
    else
        @warn "suppressed_tasks.jdl2 not found in $model_dir, skipping suppression-state restore"
    end

    println("models have been loaded")
    return (
        hive_to_source_bee = hive_to_source_bee,
        source_to_hive_bee = source_to_hive_bee,
        available_source_bees = available_source_bees,
    )
end

"""
Copy rows from `src` into `dest` using a row mapping:
- dest row i gets src row row_map[i]
- columns are copied on overlap only
"""
function overwrite_overlap_rows_cols!(
    dest::AbstractMatrix,
    src::AbstractMatrix,
    row_map::AbstractVector{<:Integer},
)
    m = min(size(dest, 1), length(row_map))
    n = min(size(dest, 2), size(src, 2))

    @inbounds for i in 1:m
        src_i = row_map[i]
        dest[i, 1:n] .= src[src_i, 1:n]
    end
    return dest
end

function run_single_simulation(config::Dict, output_dir::String, foldername::String; 
                              timestamp::Bool=false, verbose::Bool=true, save_results=false, save_all=false)
    """Run a single simulation with given configuration"""
    
    println("ouput_dir: ", output_dir)
    
    # Make a copy to avoid modifying the original config
    config = copy(config)
    
    timepoint = now()
    config["run_timestamp"] = string(timepoint)
    
    # Handle random seed
    if get(config, "seed", nothing) === nothing
        # Generate a seed based on current time
        seed_value = abs(hash(time_ns())) % Int32
        config["seed"] = Int(seed_value)
    else
        seed_value = Int(config["seed"])
    end
    
    # Set the random seed for reproducibility
    Random.seed!(seed_value)
    
    if verbose
        println("Starting simulation with configuration:")
        for (key, value) in config
            println("  $key: $value")
        end
        println("Random seed: $(seed_value)")
        println()
    end
    
    # Initialize loaders and hive
    if get(config,"use_gauss_dataset", false) === true
        gauss_dataset_dir = config["gauss_dataset_dir"]
        gauss_dataset_name = config["gauss_dataset_name"]
        gauss_dataset_path = joinpath(gauss_dataset_dir, gauss_dataset_name)
        memtag("Before loading Gaussian dataset")
        #all_datasets = load_gauss_dataset(gauss_dataset_path)
        #dataset_conf = load_gauss_metadata(gauss_dataset_dir)
        #n_total = length(all_datasets)

        ## Reproducible RNG for task selection
        #rng = haskey(config, "seed") ? MersenneTwister(config["seed"]) : Random.default_rng()

        #mode = get(config, "task_selection_mode", "first") |> Symbol   # "first" or "random"
        #task_ids = select_task_indices(config["n_tasks"], n_total; mode=mode, rng=rng)

        #config["task_ids"] = task_ids
        #config["dataset_names"] = Symbol.("task_$(i)" for i in task_ids)

        #println("Selected task_ids: ", config["task_ids"])
        #println("Dataset names: ", config["dataset_names"])

        #memtag("After loading Gaussian dataset")
        #loaders = prepare_all_gauss_loaders(all_datasets; batchsize=config["batch_size"], shuffle_train=true)
        #memtag("After preparing Gaussian data loaders")

        #model_template = create_gauss_model_template(dataset_conf["features_dimension"], dataset_conf["n_classes"])

        all_datasets = load_gauss_dataset(gauss_dataset_path)
        dataset_conf = load_gauss_metadata(gauss_dataset_dir)
        n_total = length(all_datasets)

        rng = haskey(config, "seed") ? MersenneTwister(config["seed"]) : Random.default_rng()

        # sets config["task_ids"] and config["dataset_names"], with restart logic
        choose_tasks_with_restart_support!(config, n_total; rng=rng)

        loaders = prepare_all_gauss_loaders(all_datasets; batchsize=config["batch_size"], shuffle_train=true)
        model_template = create_gauss_model_template(dataset_conf["features_dimension"], dataset_conf["n_classes"])

    elseif get(config, "use_gauss", false) === false
        loaders, task_info, model_template = prepare_multitask_setup(config["dataset_names"]; 
                                                batch_size=config["batch_size"])
    else
        variance_bounds = (Float64(config["variance_bounds"][1]), Float64(config["variance_bounds"][2]))
        center_generation_bounds = (Float64(config["center_generation_bounds"][1]), Float64(config["center_generation_bounds"][2]))
        conf = TaskConfig(
            config["n_classes"],
            config["n_tasks"],
            config["features_dimension"],
            config["n_per_class_train"],
            config["n_per_class_test"],
            config["use_per_class_variance"],
            variance_bounds,
            center_generation_bounds
        )
        dataset = create_dataset(conf)
        all_datasets = generate_rotated_tasks(dataset, conf.n_tasks)
        loaders = prepare_all_gauss_loaders(all_datasets; batchsize=config["batch_size"], shuffle_train=true)
        model_template = create_gauss_model_template(conf.features_dimension, conf.n_classes)
        config["dataset_names"] = Symbol.("task_$(i)" for i in 1:conf.n_tasks)
    end

    hive = initialize_hive_from_config(config, model_template)
    
    load_info = nothing

    if get(config, "load_models_from", "") != ""
        model_dir = config["load_models_from"]
        println("Loading initial models from directory: $model_dir")

        # Optional reproducible seed from config
        rng =
            haskey(config, "seed") ?
            MersenneTwister(Int(config["seed"])) :
            Random.default_rng()

        load_info = load_models_into_hive!(hive, model_dir; rng=rng, shuffle_if_subset=true)

        # You can keep this in memory for downstream logic and save it for stitching
        println("Loaded bee mapping (hive -> source): ", load_info.hive_to_source_bee)
    end

    if load_info !== nothing
        mapping_path = joinpath(output_dir, foldername, "loaded_bee_mapping.jdl2")
        mkpath(dirname(mapping_path))
        JLD2.@save mapping_path load_info
    end
    
    println("Starting simulation...")
    # Run simulation
    start_time = time()
    results = run_gillespie_simulation!(hive, loaders, output_dir, foldername; verbose=verbose)

    end_time = time()
    
    if verbose
        println("Simulation completed:")
        println("  Final time: $(results.final_time)")
        println("  Total events: $(results.total_events)")
        println("  Run time: $(round(end_time - start_time, digits=2)) seconds")
        println()
    end

    config["run_time"] = round(end_time - start_time, digits=2)

    run_output_dir = joinpath(output_dir, foldername)
    println("Run output directory: $run_output_dir")
    if get(config, "save_nn_epochs", 0) > 0 && length(results.model_states) > 0
        for (i, state_snapshot) in enumerate(results.model_states)
            epoch = state_snapshot.epoch
            time = state_snapshot.time
            models = state_snapshot.models
            epoch_dir = joinpath(run_output_dir, "epoch_$(epoch)")
            mkpath(epoch_dir)
            for (b, model_state) in enumerate(models)
                model_path = joinpath(epoch_dir, "bee_$(b)_model.jdl2")
                JLD2.@save model_path model_state
                #jdlsave(model_path; model_state)
                # Save model state using BSON
                #using BSON
                #BSON.@save model_path model_state
            end
            JLD2.@save joinpath(epoch_dir, "suppressed_tasks.jdl2") suppressed_tasks=state_snapshot.suppressed_tasks suppression_time_left=state_snapshot.suppression_time_left
        end    
    end
    
    if !save_results
        return results
    end

    if timestamp
        time_str = Dates.format(timepoint, "yyyy-mm-dd_HHMMSS")
        foldername *= "_$time_str"
    end

    
    if save_all
        save_simulation_results(results, run_output_dir;
                                save_states=true,
                                save_activity=true,
                                save_events=true,
                                save_losses=false)
    else
        println("Saving summary results...")
        save_simulation_results(results, run_output_dir)
        println("Summary results saved to: $run_output_dir")
    end

    save_log = false
    if save_log == true
        log_df = log_to_dataframe(results.log)
        println("Saving event log (raw and DataFrame)...")
        println("log DataFrame size: ", size(log_df))
        println("short view:")
        show(first(log_df, 5))

        try
            save_log_df(log_df, run_output_dir)
        catch e
            @warn "Could not save event log DataFrame: $e"
        end
    end

    # Save configuration and metadata
    save_metadata_to_config(config, run_output_dir)
    task_mapping = Dict(k => string(v) for (k, v) in hive.config.index_to_task_mapping)
    save_task_mapping(task_mapping, run_output_dir)

    if verbose
        println("All results saved to directory: $run_output_dir")
    end

    return results
end

function main()

    """Main function for command line usage"""
    args = parse_commandline()
    if !isempty(args["config"])
        config = load_config_from_json(args["config"])
    else
        config = create_default_config()
    end
    
    # Merge with command line arguments
    config = merge_config_with_args(config, args)

    if get(config, "use_gauss_dataset", false) === true
        # For Gaussian dataset tasks, set max dimensions based on loaded dataset
        gauss_dataset_dir = config["gauss_dataset_dir"]
        dataset_conf = load_gauss_metadata(gauss_dataset_dir)
        config["max_input_dim"] = dataset_conf["features_dimension"]
        config["max_output_dim"] = dataset_conf["n_classes"]
    elseif get(config, "use_gauss", false) === false
        config["dataset_names"] = Symbol.(config["dataset_names"])
        max_input_dim, max_output_dim = calculate_universal_dimensions(config["dataset_names"])
        config["max_input_dim"] = max_input_dim
        config["max_output_dim"] = max_output_dim
    else
        # For Gaussian tasks, set max dimensions based on config
        config["max_input_dim"] = config["features_dimension"]
        config["max_output_dim"] = config["n_classes"]
    end

    #println("Running simulation with the following configuration:")
    #for (key, value) in config
        #println("  $key: $value")
    #end
    #println()

    # Run simulation
    
    # Saving in alternative directory structure
    println("args outputdir: ", args["output-dir"])
    folder_name = basename(dirname(args["output-dir"]))
    alt_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint"
    alt_output_dir = joinpath(alt_dir, folder_name, "data")
    println("alt output dir: ", alt_output_dir)

    results = run_single_simulation(
        config,
        #args["output-dir"],
        alt_output_dir,
        args["base-name"];
        timestamp=args["timestamp"],
        verbose=args["verbose"],
        save_results=args["save-results"],
        save_all=args["save-all"]
    )
    
    println("Simulation completed successfully!")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end