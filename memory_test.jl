#!/usr/bin/env julia

"""
Memory monitoring test script for the bee simulation.
Runs a reduced simulation while logging memory usage to identify bottlenecks.
"""

using Pkg
Pkg.activate("./env_nethive_multiverse")

using JSON3
using Random
using Dates
using CSV
using DataFrames
using Flux

# Include simulation modules
include("src/core/definitions.jl")
include("src/data/registry.jl") 
include("src/data/loaders.jl")    # For prepare_multitask_setup
include("src/core/multitask_training.jl")
include("src/core/methods.jl")
include("src/core/save_data.jl")
include("run_simulation.jl")  # For initialize_hive_from_config

function log_memory_usage(stage::String, start_time::Float64)
    """Log current memory usage with timestamp"""
    current_time = time()
    elapsed = current_time - start_time
    
    # Get memory info
    total_mem = Sys.total_memory() / 1e9  # GB
    free_mem = Sys.free_memory() / 1e9    # GB 
    used_mem = total_mem - free_mem
    
    # Get process memory (RSS)
    try
        rss_kb = parse(Int, readchomp(`ps -o rss= -p $(getpid())`))
        rss_gb = rss_kb / 1e6  # Convert KB to GB
        println("[$stage] Time: $(round(elapsed, digits=1))s | Total: $(round(total_mem, digits=1))GB | Used: $(round(used_mem, digits=1))GB | Process RSS: $(round(rss_gb, digits=2))GB")
    catch
        println("[$stage] Time: $(round(elapsed, digits=1))s | Total: $(round(total_mem, digits=1))GB | Used: $(round(used_mem, digits=1))GB | Process RSS: unknown")
    end
    
    # Force garbage collection and log again
    GC.gc()
    sleep(0.1)  # Let GC finish
    try
        rss_kb = parse(Int, readchomp(`ps -o rss= -p $(getpid())`))
        rss_gb = rss_kb / 1e6
        println("[$stage] After GC: Process RSS: $(round(rss_gb, digits=2))GB")
    catch
        println("[$stage] After GC: Process RSS: unknown")
    end
end

function run_memory_test()
    """Run a minimal simulation with memory monitoring"""
    
    start_time = time()
    println("=== Memory Test Started at $(Dates.now()) ===")
    log_memory_usage("STARTUP", start_time)
    
    # Minimal test configuration
    config = Dict(
        "n_bees" => 10,           # Reduced from typical 50-100
        "n_epochs" => 5,          # Reduced from typical 100+
        "batch_size" => 32,       # Small batch size
        "learning_rate" => 0.001,
        "production_rate" => 1.0,
        "interaction_rate" => 0.1,
        "punish_rate" => 0.01,
        "lambda_sensitivity" => 0.1,
        "dataset_names" => ["mnist", "fashion_mnist"],  # Only 2 tasks
        "max_input_dim" => 784,
        "max_output_dim" => 10,
        "n_steps_per_epoch" => 10,  # Very reduced
        "save_nn_epochs" => 0,      # No NN saving to reduce memory
        "random_seed" => 42
    )
    
    println("Config: $(length(config["dataset_names"])) tasks, $(config["n_bees"]) bees, $(config["n_epochs"]) epochs")
    log_memory_usage("CONFIG_LOADED", start_time)
    
    # Set random seed
    Random.seed!(config["random_seed"])
    log_memory_usage("SEED_SET", start_time)
    
    # Load datasets using the same method as run_simulation.jl
    println("Loading datasets...")
    dataset_symbols = [Symbol(name) for name in config["dataset_names"]]
    loaders, task_info, model_template = prepare_multitask_setup(dataset_symbols; 
                                            batch_size=config["batch_size"])
    log_memory_usage("DATASETS_LOADED", start_time)
    
    # Initialize hive using the same method as run_simulation.jl
    println("Initializing hive...")
    hive = initialize_hive_from_config(config, model_template)
    log_memory_usage("HIVE_INITIALIZED", start_time)
    
    # Setup output directory
    output_dir = "memory_test_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))"
    mkpath(output_dir)
    
    # Save initial config
    open(joinpath(output_dir, "config.json"), "w") do f
        JSON3.pretty(f, config)
    end
    log_memory_usage("OUTPUT_SETUP", start_time)
    
    # Run a very simplified simulation with memory monitoring
    println("Starting simplified simulation...")
    
    # Just run a few Gillespie steps to see memory usage
    for step in 1:20  # Only 20 total steps for the whole test
        println("  Step $step/20")
        
        # Run one step and log memory
        event_occurred, selected_action = gillespie_step!(hive, loaders)
        
        if !event_occurred
            println("  No more events possible, stopping early")
            break
        end
        
        # Log memory every 5 steps
        if step % 5 == 0
            log_memory_usage("STEP_$(step)", start_time)
        end
    end
    
    log_memory_usage("SIMULATION_COMPLETE", start_time)
    
    # Create simple results to test saving
    println("Creating test results for saving...")
    dummy_results = (
        production_count = zeros(Int, 2, hive.config.n_bees, hive.config.n_tasks),
        suppression_count = zeros(Int, 2, hive.config.n_bees, hive.config.n_tasks), 
        performance_history = hive.queen_genes,
        loss_history = hive.losses,
        final_time = hive.current_time,
        total_events = 10
    )
    log_memory_usage("DUMMY_RESULTS_CREATED", start_time)
    
    # Test saving
    save_simulation_results(dummy_results, output_dir; save_states=true, save_events=false, save_losses=false)
    log_memory_usage("RESULTS_SAVED", start_time)
    
    total_time = time() - start_time
    println("\\n=== Memory Test Completed ===")
    println("Total runtime: $(round(total_time, digits=1)) seconds")
    println("Output saved to: $output_dir")
    log_memory_usage("COMPLETE", start_time)
    
    return output_dir
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting memory monitoring test...")
    try
        output_dir = run_memory_test()
        println("SUCCESS: Memory test completed. Check output in: $output_dir")
        exit(0)
    catch e
        println("ERROR: Memory test failed with: $e")
        println(stacktrace())
        exit(1)
    end
end