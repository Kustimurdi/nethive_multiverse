"""
Multi-Task Gillespie Simulation Runner

This module provides the main simulation loop that orchestrates the multi-task Gillespie 
algorithm over multiple epochs.
"""

using Logging
using Statistics

include("multitask_gillespie.jl")

"""
    run_multitask_simulation!(hive::MultiTaskHive, train_loaders::Dict, test_loaders::Dict;
                              verbose::Bool=true, save_frequency::Int=0)

Run the complete multi-task Gillespie simulation for the specified number of epochs.

# Arguments
- `hive::MultiTaskHive`: Initial hive state (will be modified)
- `train_loaders::Dict`: Training data loaders by dataset name
- `test_loaders::Dict`: Test data loaders by dataset name
- `verbose::Bool=true`: Whether to log progress information
- `save_frequency::Int=0`: How often to save state (0 = no saving)

# Returns
- `simulation_log::Vector{Dict}`: Complete log of all events and results
- `epoch_summaries::Vector{Dict}`: Summary statistics for each epoch

# Side Effects
- Modifies hive state throughout simulation
- Updates current_epoch and current_time in hive
- Logs progress if verbose=true
"""
function run_multitask_simulation!(hive::MultiTaskHive, train_loaders::Dict, test_loaders::Dict;
                                   verbose::Bool=true, save_frequency::Int=0)
    
    config = hive.config
    simulation_log = Dict{String, Any}[]
    epoch_summaries = Dict{String, Any}[]
    
    if verbose
        @info "Starting multi-task Gillespie simulation" n_bees=hive.n_bees n_tasks=hive.n_tasks n_epochs=config.n_epochs
    end
    
    # Initialize performance baselines
    if verbose
        @info "Computing initial performance baselines..."
        update_all_queen_genes!(hive, test_loaders)
        initial_summary = get_hive_performance_summary(hive)
        @info "Initial performance" mean_per_task=initial_summary["mean_performance_per_task"] overall_mean=initial_summary["overall_mean_performance"]
    end
    
    total_events = 0
    total_train_events = 0
    total_interact_events = 0
    
    # Main simulation loop over epochs
    for epoch in 1:config.n_epochs
        epoch_start_time = time()
        hive.current_epoch = UInt32(epoch)
        epoch_target_time = Float64(epoch)
        
        epoch_events = 0
        epoch_train_events = 0 
        epoch_interact_events = 0
        epoch_log = Dict{String, Any}[]
        
        if verbose
            @info "Starting epoch $epoch" target_time=epoch_target_time current_sim_time=hive.current_time
        end
        
        # Run Gillespie steps until we reach the epoch target time
        while hive.current_time < epoch_target_time
            
            # Execute one Gillespie step
            event, result = multitask_gillespie_step!(hive, train_loaders, test_loaders)
            
            # Log the event
            event_log = Dict(
                "epoch" => epoch,
                "event_id" => total_events + 1,
                "timestamp" => event.timestamp,
                "event_type" => event.event_type,
                "bee_idx" => event.bee_idx,
                "task_idx" => event.task_idx,
                "partner_bee_idx" => event.partner_bee_idx,
                "result" => result
            )
            push!(epoch_log, event_log)
            push!(simulation_log, event_log)
            
            # Update counters
            epoch_events += 1
            total_events += 1
            
            if event.event_type == :train
                epoch_train_events += 1
                total_train_events += 1
            elseif event.event_type == :interact
                epoch_interact_events += 1
                total_interact_events += 1
            end
            
            # Safety check to prevent infinite loops
            if epoch_events > config.n_steps_per_epoch * 10  # 10x safety margin
                @warn "Epoch $epoch exceeded safety limit of events, advancing to next epoch"
                break
            end
        end
        
        # Epoch summary and evaluation
        epoch_elapsed_time = time() - epoch_start_time
        
        # Update all queen genes for summary
        update_all_queen_genes!(hive, test_loaders)
        performance_summary = get_hive_performance_summary(hive)
        
        epoch_summary = Dict(
            "epoch" => epoch,
            "elapsed_time" => epoch_elapsed_time,
            "simulation_time" => hive.current_time,
            "n_events" => epoch_events,
            "n_train_events" => epoch_train_events,
            "n_interact_events" => epoch_interact_events,
            "train_ratio" => epoch_train_events / max(epoch_events, 1),
            "interact_ratio" => epoch_interact_events / max(epoch_events, 1),
            "performance_summary" => performance_summary
        )
        push!(epoch_summaries, epoch_summary)
        
        if verbose
            @info "Epoch $epoch completed" elapsed_time=round(epoch_elapsed_time, digits=2) n_events=epoch_events n_train=epoch_train_events n_interact=epoch_interact_events mean_performance=round(performance_summary["overall_mean_performance"], digits=4)
        end
        
        # Optional: save state periodically
        if save_frequency > 0 && epoch % save_frequency == 0
            if verbose
                @info "Saving state at epoch $epoch"
            end
            # TODO: Implement state saving functionality
        end
    end
    
    # Final summary
    if verbose
        final_summary = get_hive_performance_summary(hive)
        @info "Multi-task simulation completed" total_events=total_events total_train=total_train_events total_interact=total_interact_events final_mean_performance=round(final_summary["overall_mean_performance"], digits=4)
        
        # Print per-task final performance
        for (i, task_name) in enumerate(hive.config.dataset_names)
            task_performance = final_summary["mean_performance_per_task"][i]
            @info "Final performance on $task_name" accuracy=round(task_performance, digits=4)
        end
    end
    
    return simulation_log, epoch_summaries
end

"""
    create_multitask_simulation_summary(simulation_log::Vector{Dict}, epoch_summaries::Vector{Dict})

Create a comprehensive summary of the simulation results.

# Arguments
- `simulation_log::Vector{Dict}`: Complete event log from simulation
- `epoch_summaries::Vector{Dict}`: Per-epoch summary statistics

# Returns
- `summary::Dict`: Comprehensive simulation summary with statistics and analysis
"""
function create_multitask_simulation_summary(simulation_log::Vector{Dict}, epoch_summaries::Vector{Dict})
    
    total_events = length(simulation_log)
    total_epochs = length(epoch_summaries)
    
    # Event type analysis
    train_events = [e for e in simulation_log if e["event_type"] == :train]
    interact_events = [e for e in simulation_log if e["event_type"] == :interact]
    
    # Performance trajectory
    performance_trajectory = [es["performance_summary"]["overall_mean_performance"] for es in epoch_summaries]
    initial_performance = length(performance_trajectory) > 0 ? performance_trajectory[1] : 0.0
    final_performance = length(performance_trajectory) > 0 ? performance_trajectory[end] : 0.0
    
    # Training distribution analysis
    if !isempty(train_events)
        # Count training events per bee
        train_by_bee = Dict{Int, Int}()
        train_by_task = Dict{Int, Int}()
        
        for event in train_events
            bee_idx = event["bee_idx"]
            task_idx = event["task_idx"]
            train_by_bee[bee_idx] = get(train_by_bee, bee_idx, 0) + 1
            train_by_task[task_idx] = get(train_by_task, task_idx, 0) + 1
        end
    else
        train_by_bee = Dict{Int, Int}()
        train_by_task = Dict{Int, Int}()
    end
    
    summary = Dict(
        "total_events" => total_events,
        "total_epochs" => total_epochs,
        "total_train_events" => length(train_events),
        "total_interact_events" => length(interact_events),
        "train_ratio" => length(train_events) / max(total_events, 1),
        "interact_ratio" => length(interact_events) / max(total_events, 1),
        "performance_trajectory" => performance_trajectory,
        "initial_performance" => initial_performance,
        "final_performance" => final_performance,
        "performance_improvement" => final_performance - initial_performance,
        "train_distribution_by_bee" => train_by_bee,
        "train_distribution_by_task" => train_by_task,
        "epoch_summaries" => epoch_summaries
    )
    
    return summary
end