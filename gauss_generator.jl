using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/env_nethive_multiverse")
#Pkg.instantiate()

# Load packages directly (they should be available via JULIA_PROJECT)
using ArgParse
using CSV
using DataFrames
using Distributions
using JSON3
using LinearAlgebra
using Random
using Statistics
using Dates
using JLD2
using Flux
#using BSON: @save, @load

# Load our modules
include("src/data/synthetic.jl")
#include("src/data/loaders.jl")
include("src/core/definitions.jl")
#include("src/core/multitask_training.jl")
#include("src/core/methods.jl")
include("src/core/save_data.jl")

file_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_datasets"
name = now()
name = Dates.format(name, "yyyymmddHHMMSS")
file_path = string(file_dir, "/gauss_dataset_", name, ".jld2")

config = Dict(
    "n_classes" => 15,
    "n_tasks" => 15,
    "features_dimension" => 10,
    "n_per_class_train" => 100,
    "n_per_class_test" => 50,
    "use_per_class_variance" => true,
    "variance_bounds" => [0.01, 0.05],
    "center_generation_bounds" => [0.0, 1.0],
    "seed" => "nothing"
)
variance_bounds = (Float64(config["variance_bounds"][1]), Float64(config["variance_bounds"][2]))
center_generation_bounds = (Float64(config["center_generation_bounds"][1]), Float64(config["center_generation_bounds"][2]))
taskconfig = TaskConfig(
    config["n_classes"],
    config["n_tasks"],
    config["features_dimension"],
    config["n_per_class_train"],
    config["n_per_class_test"],
    config["use_per_class_variance"],
    variance_bounds,
    center_generation_bounds
)

dataset = create_dataset(taskconfig)
all_datasets = generate_rotated_tasks(dataset, taskconfig.n_tasks)
if !isdir("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_datasets")
    mkpath("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/gauss_datasets")
end
@save file_path all_datasets
save_metadata_to_config(config, file_dir)