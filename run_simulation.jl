using Pkg
Pkg.activate("./env_nethive_multiverse/")
#Pkg.instantiate()

# Load packages directly (they should be available via JULIA_PROJECT)
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

# Load our modules
include("src/data/loaders.jl")
include("src/data/synthetic.jl")
include("src/core/definitions.jl")
include("src/core/multitask_training.jl")
include("src/core/methods.jl")
include("src/core/save_data.jl")

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

function run_single_simulation(config::Dict, output_dir::String, foldername::String; 
                              timestamp::Bool=false, verbose::Bool=true, save_results=false, save_all=false)
    """Run a single simulation with given configuration"""
    
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

    if get(config, "use_gauss", false) === false
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
            #Tuple(config["variance_bounds"]),
            #Tuple(config["center_generation_bounds"]),
            #config["variance_bounds"],
            #config["center_generation_bounds"],
            variance_bounds,
            center_generation_bounds
        )
        loaders = prepare_all_gauss_loaders(conf; batchsize=config["batch_size"], shuffle_train=true)
        model_template = create_gauss_model_template(conf)
        config["dataset_names"] = Symbol.("task_$(i)" for i in 1:conf.n_tasks)
        
        println("Gaussian loaders prepared for $(length(loaders)) tasks.")
        println("Each task has $(length(first(values(loaders))["train"])) training samples.")
        batchsize = config["batch_size"]
        println("Using batch size: $batchsize")
        for (k, v) in loaders
            println("Loader for $(k):")
            println("  Train size: $(length(v["train"]))")
            println("  Test size: $(length(v["test"]))")
            #@show size(x_batch) size(y_batch) # Debugging output
            #@show typeof(x_batch) typeof(y_batch) # Debugging output
            #@show size(model(x_batch)) # Debugging output
            #@show size(y_batch) # Debugging output
        end
    end

    hive = initialize_hive_from_config(config, model_template)
    
    println("Starting simulation...")
    # Run simulation
    start_time = time()
    results = run_gillespie_simulation!(hive, loaders; verbose=verbose)

    end_time = time()
    
    if verbose
        println("Simulation completed:")
        println("  Final time: $(results.final_time)")
        println("  Total events: $(results.total_events)")
        println("  Wall time: $(round(end_time - start_time, digits=2)) seconds")
        println()
    end
    
    if !save_results
        return results
    end

    if timestamp
        time_str = Dates.format(timepoint, "yyyy-mm-dd_HHMMSS")
        foldername *= "_$time_str"
    end

    run_output_dir = joinpath(output_dir, foldername)
    
    if save_all
        save_simulation_results(results, run_output_dir;
                                save_states=true,
                                save_events=true,
                                save_losses=true)
    else        
        #save_simulation_results(results, run_output_dir;
                                #save_states=true,
                                #save_events=true,
                                #save_losses=false)
    end

    log_df = log_to_dataframe(results.log)
    println("Saving event log (raw and DataFrame)...")
    println("log DataFrame size: ", size(log_df))
    println("short view:")
    show(first(log_df, 5))
    # Save raw GillespieEventLog as JSON (if available) and also save the DataFrame CSV
    #try
        #save_log(results.log, run_output_dir)
    #catch e
        #@warn "Could not save raw event log: $e"
    #end

    try
        save_log_df(log_df, run_output_dir)
    catch e
        @warn "Could not save event log DataFrame: $e"
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

    if get(config, "use_gauss", false) === false
        config["dataset_names"] = Symbol.(config["dataset_names"])
        max_input_dim, max_output_dim = calculate_universal_dimensions(config["dataset_names"])
        config["max_input_dim"] = max_input_dim
        config["max_output_dim"] = max_output_dim
    else
        # For Gaussian tasks, set max dimensions based on config
        config["max_input_dim"] = config["features_dimension"]
        config["max_output_dim"] = config["n_classes"]
    end

    println("Running simulation with the following configuration:")
    for (key, value) in config
        println("  $key: $value")
    end
    println()

    # Run simulation
    results = run_single_simulation(
        config,
        args["output-dir"],
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
