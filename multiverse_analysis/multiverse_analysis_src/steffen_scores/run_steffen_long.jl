using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_analysis/env_multiverse_analysis")

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
    #"/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/lr5-6/2b2t15c10000e5-6lr100ls_full_phase_diagram",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/lr5-6/2b2t15c10000e5-6lr_dead_time",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/lr5-6/2b2t15c10000e5-6lr_int_rate"

    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_sensitivity_sweep"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_rechts",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_unten"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/3b3t2000e_car_iris_wine_sweep"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/3b3t_car_wine_bank"

    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/3b3t_car_bank_wdbc"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/3b3t_car_bank_wdbc"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t15c10000e5-6lr100ls_high_dt"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_vert_rechts"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_vert_rechts_second_half"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/2b2t5-6lr_missing_runs"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_vert_rechts_second_half"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/50b50t"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t5-6lr_heaviside",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t_heaviside2",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t_heaviside3"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t_heaviside5",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t_heaviside6"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t5-6lr_barebone"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t100ls"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t100ls_add1",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t100ls_add2"
    #"/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/3b3t_heaviside"
    "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_heaviside2/",
    "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_heaviside3/",
]

println("is")
results, failures = steffen_scores_all_runs_from_config_long(roots; thresholds=[0.8, 0.9])

println("good")
#output_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/analysis_out"
#output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_vert_rechts"
#output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_eta_sweep_vert_rechts_second_half"
#output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/2b2t5-6lr_missing_runs"
#output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t100ls"
#output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/10b10t100ls"
output_dir = "/project/theorie/n/N.Pfaffenzeller/results_project/checkpoint/5b5t_heaviside"

prefix = "steffen_scores_5b5t_heaviside_2_3_$(Dates.format(now(), "yyyy-mm-dd_HHMM"))"
paths = save_analysis_outputs(results, failures,
    output_dir;
    prefix=prefix
)

println("Saved: ", paths)
println("OK runs: ", nrow(results), " | failures: ", nrow(failures))
