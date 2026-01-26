using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_analysis/env_mutltiverse_analysis")

using DataFrames
using CSV
using JSON3
using Dates
using Statistics
using ProgressMeter
using Base.Threads

include("../steffen_scores/steffen_score_long.jl")
println("what")

roots = [
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/2b2t15c10000e5-6lr100ls_full_phase_diagram",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/2b2t15c10000e5-6lr_dead_time",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/2b2t15c10000e5-6lr_int_rate"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_sensitivity_sweep"
    "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_rechts"
]

println("is")
results, failures = steffen_scores_all_runs_from_config_long(roots; thresholds=[0.8, 0.9])

println("good")
output_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/analysis_out"
prefix = "steffen_scores_5b5t_eta_sweep_$(Dates.format(now(), "yyyy-mm-dd_HHMM"))"
#prefix = "steffen_test"
paths = save_analysis_outputs(results, failures,
    output_dir;
    prefix=prefix
)

println("Saved: ", paths)
println("OK runs: ", nrow(results), " | failures: ", nrow(failures))
