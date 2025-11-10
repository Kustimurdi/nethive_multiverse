# Smoke test for log_events and push_event!
# This script is intended to be lightweight: it includes the core definitions and methods
# and exercises log_events to ensure events are recorded correctly.

using Flux

# Include core files (these files may depend on other files in the project; adjust paths if needed)
include(joinpath(@__DIR__, "../src/core/definitions.jl"))
include(joinpath(@__DIR__, "../src/core/methods.jl"))

# Create a minimal model template: simple Dense network matching input/output dims
model_template = () -> Chain(Dense(1, 2), identity)

# Build a minimal config matching the MultiTaskHiveConfig constructor
dataset_names = [:dummy1, :dummy2]
max_input_dim = 1
max_output_dim = 2
n_bees = 2
n_epochs = 1
n_steps_per_epoch = 10
production_rate = 1.0
interaction_rate = 0.5
learning_rate = 0.001
punish_rate = 0.01
lambda_sensitivity = 1.0
random_seed = 42
save_nn_epochs = 1
batches_per_step = nothing
dead_time = 1.0

config = MultiTaskHiveConfig(dataset_names, model_template, max_input_dim, max_output_dim,
                             n_bees, n_epochs, n_steps_per_epoch, production_rate,
                             interaction_rate, learning_rate, punish_rate, lambda_sensitivity,
                             random_seed, save_nn_epochs, batches_per_step, dead_time)

# Create hive
hive = MultiTaskHive(config)

# Initialize some queen genes for testing
hive.queen_genes .= [0.5 0.6; 0.4 0.7]

# Create a log and an action
log = GillespieEventLog()
action = (type=:produce, bee1=1, bee2=1, task=1)

# Call log_events
println("Calling log_events...")
log_events(log, hive, action)

println("Log contents after one event:")
@show log.time
@show log.bee1_id
@show log.bee2_id
@show log.task_id
@show log.accuracies

# Validate that the recorded accuracy matches hive.queen_genes for bee1
recorded = log.accuracies[1]
expected = hive.queen_genes[1, :]
println("Recorded accuracies: ", recorded)
println("Expected accuracies: ", expected)

if all(isapprox.(recorded, expected; atol=1e-8))
    println("Smoke test PASSED: log_events recorded accuracies correctly.")
    exit(0)
else
    println("Smoke test FAILED: mismatch in recorded accuracies.")
    exit(2)
end
