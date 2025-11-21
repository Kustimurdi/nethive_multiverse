# punishment type is now part of MultiTaskHiveConfig (hive.config.punishment)

function get_production_rates_basic(hive::MultiTaskHive)::Matrix{Float64}
    """
    Calculate production rates for all bees and tasks.

    Returns a matrix where result[bee, task] represents the production rate
    of the given bee for the given task.
    """
    return hive.config.production_rate .* ones(Float64, hive.n_bees, hive.n_tasks)
end

function get_production_rates_time_out(hive::MultiTaskHive)::Matrix{Float64}
    """
    Calculate production rates for all bees and tasks in the hive.

    Returns a matrix where result[bee, task] represents the production rate
    of the given bee for the given task.
    """
    production_rates = hive.config.production_rate * (.!hive.suppressed_tasks)
    return production_rates
end

"""
    interaction_kernel(bee1_task_values, bee2_task_values, target_task, config)

Calculate interaction kernel between two bees for a specific task.

Returns the strength of influence that bee2 has on bee1 for the target task.
"""
function interaction_kernel(bee1_task_values::Vector{Float64}, bee2_task_values::Vector{Float64}, 
                           target_task::Int, lambda_sensitivity)::Float64
    # Get the calculation functions from the dictionaries
    prefactor_fn = (b1::Vector{Float64}, b2::Vector{Float64}, t::Int) -> b1[t] * b2[t]
    sigmoid_arg_fn = (b1::Vector{Float64}, b2::Vector{Float64}, t::Int) -> b2[t] - b1[t]
    # Calculate components
    prefactor = prefactor_fn(bee1_task_values, bee2_task_values, target_task)
    sigmoid_arg = sigmoid_arg_fn(bee1_task_values, bee2_task_values, target_task)
    
    # Apply sigmoid transformation
    sigmoid_factor = 1.0 / (1.0 + exp(-lambda_sensitivity * sigmoid_arg))
    
    return prefactor * sigmoid_factor
end

function get_interaction_rates(performance_values::Matrix{Float64}, interaction_rate::Float64, 
                                lambda_sensitivity::Float64)::Array{Float64, 3}
    """
    Calculate individual interaction rates for all bee pairs and tasks.
    
    Returns a 3D array where result[bee_i, bee_j, task_k] represents the interaction rate
    from bee_j influencing bee_i on task_k.
    
    Dimensions: (n_bees, n_bees, n_tasks)
    - First dimension: target bee (being influenced)
    - Second dimension: source bee (providing influence) 
    - Third dimension: task being influenced
    """
    n_bees, n_tasks = size(performance_values)
    interaction_rates = zeros(Float64, n_bees, n_bees, n_tasks)
    
    # For each target bee (being influenced)
    for bee_i in 1:n_bees
        bee_i_tasks = performance_values[bee_i, :]
        
        # For each source bee (providing influence)
        for bee_j in 1:n_bees
            if bee_j != bee_i  # No self-interaction
                bee_j_tasks = performance_values[bee_j, :]
                
                # For each task
                for task_k in 1:n_tasks
                    # Calculate interaction kernel (bee_j influences bee_i on task_k)
                    kernel_value = interaction_kernel(bee_i_tasks, bee_j_tasks, task_k, lambda_sensitivity)
                    
                    # Scale by base interaction rate
                    interaction_rates[bee_i, bee_j, task_k] = interaction_rate * kernel_value
                end
            end
            # Note: interaction_rates[bee_i, bee_i, task_k] remains 0.0 (no self-interaction)
        end
    end
    
    return interaction_rates
end

function collect_all_rates(hive::MultiTaskHive)
    # Get all rate matrices
    production_rates = nothing
    pun = hive.config.punishment
    if pun == :resetting || pun == :none || pun == :gradient_ascend
        production_rates = get_production_rates_basic(hive)
    elseif pun == :time_out
        production_rates = get_production_rates_time_out(hive)
    else
        error("Unknown punishment type: $pun")
    end
    interaction_rates = get_interaction_rates(hive.queen_genes, hive.config.interaction_rate, hive.config.lambda_sensitivity)
    
    # Initialize output vectors
    rates = Float64[]
    actions = NamedTuple{(:type, :bee1, :bee2, :task), Tuple{Symbol, Int, Int, Int}}[]
    
    # Collect production rates
    for bee in 1:hive.config.n_bees, task in 1:hive.config.n_tasks
        rate = production_rates[bee, task]
        if rate > 0
            push!(rates, rate)
            push!(actions, (type=:produce, bee1=bee, bee2=bee, task=task))
        end
    end
    
    # Collect interaction (suppression) rates
    for bee1 in 1:hive.config.n_bees, bee2 in 1:hive.config.n_bees, task in 1:hive.config.n_tasks
        if bee1 != bee2  # No self-interaction
            rate = interaction_rates[bee1, bee2, task]
            if rate > 0
                push!(rates, rate)
                push!(actions, (type=:suppress, bee1=bee1, bee2=bee2, task=task))
            end
        end
    end
    
    #println("sum of production rates: $(sum(production_rates))")
    #println("sum of interaction rates: $(sum(interaction_rates))")
    
    return rates, actions
end

function execute_action!(hive::MultiTaskHive, action, loaders::Dict)
    type, bee1, bee2, task = action.type, action.bee1, action.bee2, action.task
    
    if type == :produce
        # Produce a gene for bee1, task
        dataset_name = hive.config.index_to_task_mapping[task]
        task_train_loader = loaders[dataset_name]["train"]
        task_test_loader = loaders[dataset_name]["test"]
        perform_production!(hive, bee1, task, task_train_loader, task_test_loader)
        
    elseif type == :suppress
        # bee2 (source) suppresses bee1's (target) task
        #println("Bee $bee2 suppresses Bee $bee1 on Task $task at time $(hive.current_time)")
        #println("accuracy of bee1 before suppression: $(hive.queen_genes[bee1, task])")
        #println("accuracy of bee2 before suppression: $(hive.queen_genes[bee2, task])")
        perform_suppression!(hive, bee1, bee2, task, loaders)
    else
        error("Unknown action type: $type")
    end

    accs, losses = evaluate_bee_on_all_tasks(hive, bee1, loaders) 
    hive.queen_genes[bee1, :] .= accs
    hive.losses[bee1, :] .= losses
    
    return type
end

function release_tasks!(suppressed_tasks::Array, suppression_starting_times::Array, current_time::Float64; dead_time::Float64=1.0)

    old_suppressed_tasks = copy(suppressed_tasks)
    suppressed_tasks[suppressed_tasks .& ((current_time .- suppression_starting_times) .>= dead_time)] .= false
    
    if any(old_suppressed_tasks .!= suppressed_tasks)
        #println("Released tasks at time $current_time")
        #println("Old suppressed tasks:\n$old_suppressed_tasks")
        #println("New suppressed tasks:\n$suppressed_tasks")
    end

    return nothing
end

function perform_suppression!(hive::MultiTaskHive, bee_idx::Int, partner_bee_idx::Int, task_idx::Int, loaders::Dict=nothing)
    """
    Perform an interaction event where `bee_idx` interacts with `partner_bee_idx` on `task_idx`.
    
    This function updates the hive state to reflect the interaction.
    """
    pun = hive.config.punishment
    if pun == :none
        # No action taken
        return nothing
    elseif pun == :resetting
        perform_resetting!(hive, bee_idx)
    elseif pun == :time_out
        perform_time_out!(hive, bee_idx, task_idx)
    elseif pun == :gradient_ascend
        # Not implemented yet
        perform_gradient_ascend!(hive, bee_idx, task_idx, loaders)
    else
        error("Unknown punishment type: $pun")
    end
    #perform_resetting!(hive, bee_idx)

    return nothing
end

function perform_time_out!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)
    hive.suppressed_tasks[bee_idx, task_idx] = true
    hive.suppression_start_times[bee_idx, task_idx] = hive.current_time
    return nothing
end

function perform_resetting!(hive::MultiTaskHive, bee_idx::Int)

    hive.brains[bee_idx] = hive.config.model_template()

    return nothing
end

function perform_gradient_ascend!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, loaders::Dict)
    model = hive.brains[bee_idx]
    dataloader = loaders[hive.config.index_to_task_mapping[task_idx]]["train"]
    punish_rate = hive.config.punish_rate
    punish_model!(model, dataloader, punish_rate)
    return nothing
end

function punish_model!(model, dataloader, punish_rate)
    """Apply gradient ascent to make the model worse"""
    # Create a custom optimizer that adds gradients instead of subtracting
    # We'll use Descent with a positive rate, but manually flip the gradient
    opt_state = Flux.setup(Flux.Descent(punish_rate), model)
    
    total_batch_loss = 0.0
    n_batches = 0
    
    for (x_batch, y_batch) in dataloader
        # Calculate gradients normally
        loss, grads = Flux.withgradient(model) do m
            Flux.Losses.logitcrossentropy(m(x_batch), y_batch)
        end
        
        # Manually flip the gradient to do gradient ascent
        if grads[1] !== nothing
            flipped_grads = fmap(g -> g === nothing ? nothing : -g, grads[1])  # Flip all gradients, leave nothing as nothing
            Flux.update!(opt_state, model, flipped_grads)  # This will subtract (-grad) = add grad
        end
        
        total_batch_loss += loss
        n_batches += 1
    end
    
    return total_batch_loss / max(n_batches, 1)
end

function select_gillespie_action(rates::Vector{Float64}, actions::Vector)
    # Check if any actions are available
    total_rate = sum(rates)
    if total_rate == 0.0 || length(rates) == 0
        return nothing, 0.0
    end
    
    # Draw random number from [0, total_rate)
    r = rand() * total_rate
    
    # Find which bin the random number falls into
    cumulative_rate = 0.0
    for (i, rate) in enumerate(rates)
        cumulative_rate += rate
        if r <= cumulative_rate
            return actions[i], total_rate
        end
    end
    
    # Fallback (should never reach here due to floating point precision)
    # If we somehow don't find a bin, return the last action
    return actions[end], total_rate
end

function advance_gillespie_time!(hive::MultiTaskHive, total_rate::Float64)
    if total_rate <= 0.0
        # No events possible, time doesn't advance
        return 0.0
    end
    
    # Draw from exponential distribution
    # Note: rand() gives [0,1), but we need (0,1] for log, so use 1-rand()
    # or equivalently, just use rand() since log(rand()) works fine in practice
    #dt = -log(rand()) / total_rate

    #diese version sorgt dafür, dass wir selber bestimmen, wie viele Aktionen pro Epoche geschehen
    dt = -log(rand()) / hive.config.n_steps_per_epoch
    
    # Advance the hive time
    hive.current_time += dt
    
    return dt
end

function gillespie_step!(hive::MultiTaskHive, loaders::Dict)
    #println("time for a gillespie step at time $(hive.current_time)")
    # Step 1: Collect all possible actions and their rates
    rates, actions = collect_all_rates(hive)
    
    # Step 2: Select which action to perform (returns nothing if no actions available)
    selected_action, total_rate = select_gillespie_action(rates, actions)
    
    # Check if any actions are possible
    if selected_action === nothing
        return false, nothing  # No events possible, simulation terminates
    end
    
    # Step 3: Execute the selected action
    execute_action!(hive, selected_action, loaders)
    
    # Step 4: Advance time
    advance_gillespie_time!(hive, total_rate)

    release_tasks!(hive.suppressed_tasks, hive.suppression_start_times, hive.current_time, dead_time=hive.config.dead_time)
    
    return true, selected_action  # Event occurred successfully
end

function run_gillespie_simulation!(hive::MultiTaskHive, loaders::Dict; verbose=false)

    production_count = zeros(Int, hive.config.n_epochs, hive.config.n_bees, hive.config.n_bees, hive.config.n_tasks)
    suppression_count = zeros(Int, hive.config.n_epochs, hive.config.n_bees, hive.config.n_bees, hive.config.n_tasks)
    performance_history = zeros(Float64, 1 + hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)
    loss_history = zeros(Float64, 1 + hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)

    log = GillespieEventLog()

    update_all_bees!(hive, loaders)
    for b in 1:hive.n_bees
        Main.push_event!(log, 0.0, b, 0, 0, hive.queen_genes[b, :])
    end

    performance_history[1, :, :] .= hive.queen_genes
    loss_history[1, :, :] .= hive.losses

    for epoch in 1:hive.config.n_epochs
        while hive.current_time < epoch 

            # Run one Gillespie step
            event_occurred, selected_action = gillespie_step!(hive, loaders)
            if !event_occurred
                break  # No more events possible
            end
            document_event!(selected_action, epoch, production_count, suppression_count)
            log_events(log, hive, selected_action)

            #println("action executed: $(selected_action)")
            #println("current accuracies: $(hive.queen_genes)")
            #println("current suppressed tasks: $(hive.suppressed_tasks)")
            #println()

        end

        performance_history[1+epoch, :, :] .= hive.queen_genes
        loss_history[1+epoch, :, :] .= hive.losses

        if (hive.config.save_nn_epochs) > 0 && (epoch % hive.config.save_nn_epochs == 0)
            #save the neural networks 
        end
        
        if verbose 
            println("Time: $(round(hive.current_time, digits=2)), 
            Production Events: $(sum(production_count[epoch, :,:, :])), 
            Suppression Events: $(sum(suppression_count[epoch,:, :, :]))")
        end
    end

    return (log = log, production_count = production_count, suppression_count = suppression_count, 
            performance_history = performance_history, loss_history = loss_history, 
            final_time=hive.current_time, total_events=sum(production_count) + sum(suppression_count))
end

function log_events(log::GillespieEventLog, hive::MultiTaskHive, action)
    bee_accs = hive.queen_genes[action.bee1, :]
    Main.push_event!(log, hive.current_time, action.bee1, action.bee2, action.task, bee_accs)
end

function document_event!(action, epoch::Int, production_count::Array, suppression_count::Array)
    type, bee1, bee2, task = action.type, action.bee1, action.bee2, action.task
    if type == :produce
        production_count[epoch, bee1, bee2, task] += 1
    elseif type == :suppress
        suppression_count[epoch, bee1, bee2, task] += 1
    else
        throw("this is bullshit")
    end
end