function get_production_rates(n_bees::Int, n_tasks::Int, production_rate::Float64)::Matrix{Float64}
    """
    Calculate production rates for all bees and tasks.

    Returns a matrix where result[bee, task] represents the production rate
    of the given bee for the given task.
    """
    return production_rate .* ones(Float64, n_bees, n_tasks)
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
    production_rates = get_production_rates(hive.config.n_bees, hive.config.n_tasks, hive.config.production_rate)
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
    
    println("sum of production rates: $(sum(production_rates))")
    println("sum of interaction rates: $(sum(interaction_rates))")
    
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
        perform_suppression!(hive, bee1, bee2, task)
        
    else
        error("Unknown action type: $type")
    end

    accs, losses = evaluate_bee_on_all_tasks(hive, bee1, loaders) 
    hive.queen_genes[bee1, :] .= accs
    hive.losses[bee1, :] .= losses
    
    return type
end

function perform_suppression!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, partner_bee_idx::Int)
    """
    Perform an interaction event where `bee_idx` interacts with `partner_bee_idx` on `task_idx`.
    
    This function updates the hive state to reflect the interaction.
    """
    hive.brains[bee_idx] = hive.config.model_template()

    # for now just reset the brain of the bee being influenced
    return nothing
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
    println("time for a gillespie step at time $(hive.current_time)")
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
    
    return true, selected_action  # Event occurred successfully
end

function run_gillespie_simulation!(hive::MultiTaskHive, loaders::Dict; verbose=false)

    production_count = zeros(Int, hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)
    suppression_count = zeros(Int, hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)
    performance_history = zeros(Float64, 1 + hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)
    loss_history = zeros(Float64, 1 + hive.config.n_epochs, hive.config.n_bees, hive.config.n_tasks)

    update_all_bees!(hive, loaders)
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

            accuracies, losses = evaluate_bee_on_all_tasks(hive, selected_action.bee1, loaders)
            hive.queen_genes[selected_action.bee1, :] .= accuracies
            hive.losses[selected_action.bee1, :] .= losses

            println("action executed: $(selected_action)")
            println("current accuracies: $(hive.queen_genes)")

        end

        performance_history[1+epoch, :, :] .= hive.queen_genes
        loss_history[1+epoch, :, :] .= hive.losses

        if (hive.config.save_nn_epochs) > 0 && (epoch % hive.config.save_nn_epochs == 0)
            #save the neural networks 
        end
        
        if verbose 
            println("Time: $(round(hive.current_time, digits=2)), 
            Production Events: $(sum(production_count[epoch, :, :])), 
            Suppression Events: $(sum(suppression_count[epoch, :, :]))")
        end
    end

    return (production_count = production_count, suppression_count = suppression_count, 
            performance_history = performance_history, loss_history = loss_history, 
            final_time=hive.current_time, total_events=sum(production_count) + sum(suppression_count))
end

function document_event!(action, epoch::Int, production_count::Array, suppression_count::Array)
    type, bee1, bee2, task = action.type, action.bee1, action.bee2, action.task
    if type == :produce
        production_count[epoch, bee1, task] += 1
    elseif type == :suppress
        suppression_count[epoch, bee1, task] += 1
    else
        throw("this is bullshit")
    end
end