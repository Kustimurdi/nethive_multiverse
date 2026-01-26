"""
Test script for multiverse analysis loader

Usage:
julia --project=env_mutltiverse_analysis test_loader.jl <path_to_run_directory>

Example:
julia --project=env_mutltiverse_analysis test_loader.jl ../multi_task_simulations/testruns/data/param_1_rep_1_2025-10-29_195942
"""

include("multiverse_analysis_src/load_data.jl")

function test_loader(run_path::String)
    println("Testing loader on: $run_path")
    
    if !isdir(run_path)
        error("Run directory not found: $run_path")
    end
    
    # Test individual components
    println("\n1. Testing config loading...")
    config_path = joinpath(run_path, "config.json")
    if isfile(config_path)
        config = load_config(config_path)
        println("   ✓ Config loaded: $(nrow(config)) row(s), $(ncol(config)) columns")
        println("   Columns: $(names(config))")
    else
        println("   ✗ config.json not found")
    end
    
    println("\n2. Testing states loading...")
    states_path = joinpath(run_path, "states.csv")
    if isfile(states_path)
        states = load_states(states_path)
        println("   ✓ States loaded: $(nrow(states)) rows, $(ncol(states)) columns")
        println("   Columns: $(names(states))")
        println("   Sample data:")
        println("   ", first(states, 3))
    else
        println("   ✗ states.csv not found")
    end
    
    println("\n3. Testing events loading...")
    events_path = joinpath(run_path, "events.csv")
    if isfile(events_path)
        events = load_events(events_path)
        println("   ✓ Events loaded: $(nrow(events)) rows, $(ncol(events)) columns")
        println("   Columns: $(names(events))")
    else
        println("   ⚠ events.csv not found (optional)")
    end
    
    println("\n4. Testing full run loading...")
    try
        run_data = load_run(run_path)
        println("   ✓ Full run loaded successfully")
        println("   Components: $(keys(run_data))")
    catch e
        println("   ✗ Full run loading failed: $e")
    end
    
    println("\n✅ Loader test completed!")
end

# Command line usage
if length(ARGS) > 0
    test_loader(ARGS[1])
else
    println("Usage: julia test_loader.jl <path_to_run_directory>")
    println("Example: julia test_loader.jl ../multi_task_simulations/testruns/data/param_1_rep_1_2025-10-29_195942")
end