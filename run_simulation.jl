using Pkg
Pkg.activate("./env_qg_vector/")

# Load packages directly (they should be available via JULIA_PROJECT)
using DataFrames
using CSV
using ArgParse
using JSON3
using Random
using Dates

# Load our modules
include("src/core/definitions.jl")
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
        # Initial conditions
        "--random-init"
            help = "Use random initial conditions"
            arg_type = Bool
            nargs = '?'
        "--batch-size"
            help = "Batch size for data loaders"
            arg_type = Int
            default = 32
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
        "dataset_names" => ["mnist", "fashion_mnist"],
        "n_epochs" => 10,
        "n_steps" => 100,
        "production_rate" => 500,
        "interaction_rate" => 10,
        "lambda_sensitivity" => 10.0,
        "batch_size" => 32,
        "seed" => nothing  # Will be auto-generated if not specified
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
        "batch-size" => "batch_size"
    )
    
    for (arg_key, config_key) in arg_mapping
        if haskey(args, arg_key) && (args[arg_key] !== nothing)
            merged[config_key] = args[arg_key]
        end
    end
    
    return merged
end

"""
struct MultiTaskHiveConfig
    # Multi-task fields
    dataset_names::Vector{Symbol}
    n_tasks::Int
    max_input_dim::Int
    max_output_dim::Int
    model_template::Function
    task_mapping::Dict{Symbol, Int}
    
    # Simulation parameters
    n_bees::UInt16
    n_epochs::UInt64
    n_steps_per_epoch::UInt16
    production_rate::Float64
    interaction_rate::Float64
    learning_rate::Float32
    punish_rate::Float32
    lambda_sensitivity::Float16
    random_seed::Int
    save_nn_epochs::Int
    
    # Legacy compatibility fields
    dataset_name::String
    parent_dataset_name::String
    task_config
    """

function initialize_hive_from_config(config::Dict, model_template::Function, task_info::Dict)
    """Initialize hive from configuration dictionary"""
    hive_config = MultiTaskHiveConfig(
        config["dataset_names"],
        length(config["dataset_names"]),
        config["production_rate"],
        config["degeneration_rate"],
        config["interaction_rate"],
        config["lambda_sensitivity"],
        config["time_period_suppression"],
        config["resolve_suppression_rate"],
        config["n_bees"],
        config["max_simulation_time"],
        config["n_tasks"],
        Symbol(config["interaction_kernel_prefactor_type"]),
        Symbol(config["interaction_kernel_heaviside_arg_type"]);
        record_history_interval = config["record_history_interval"]
    )
    
    hive = nHive(hive_config)
    
    # Set initial conditions
    if config["random_init"]
        println("Random initialization of hive state...")
        n_min, n_max = config["init_tasks_range"]
        hive.current_task_values .= rand(n_min:n_max, config["n_bees"], config["n_tasks"])
        hive.current_suppression_values .= rand(config["n_bees"]) .< config["init_suppression_prob"]
    end

    return hive
end

function run_single_simulation(config::Dict, output_dir::String, run_folder_name::String; 
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

    loaders, task_info, model_template = prepare_multitask_setup(dataset_names::Vector{Symbol}; batch_size::Int=32)
    max_input_dim, max_output_dim = calculate_universal_dimensions(dataset_names)
    hive = initialize_hive_from_config(config, model_template, task_info)
    
    println("Starting simulation...")
    # Run simulation
    start_time = time()
    results = run_gillespie_simulation!(hive, verbose=verbose)

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

    foldername = run_folder_name

    if timestamp
        time_str = Dates.format(timepoint, "yyyy-mm-dd_HHMMSS")
        foldername *= "_$time_str"
    end

    run_output_dir = joinpath(output_dir, foldername)
    
    if save_all
        save_simulation_results(results, run_output_dir;
                                save_states=true,
                                save_summary=true,
                                save_events=true,
                                save_suppression=true)
    else        
        save_simulation_results(results, run_output_dir;
                                save_states=true,
                                save_summary=false,
                                save_events=true,
                                save_suppression=false)
    end

    # Save configuration and metadata
    save_metadata_to_config(config, run_output_dir)

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

    println("Running simulation with the following configuration:")
    for (key, value) in config
        println("  $key: $value")
    end
    println()

    config["dataset_names"] = Symbol.(config["dataset_names"])

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