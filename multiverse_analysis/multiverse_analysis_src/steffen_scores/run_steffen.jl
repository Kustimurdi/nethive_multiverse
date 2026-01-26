using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_analysis/env_mutltiverse_analysis")

using DataFrames
using CSV
using JSON3
using Dates
using Statistics
using ProgressMeter
using Base.Threads

include("../steffen_scores/steffen_score.jl")
include("../steffen_scores/steffen_score_long.jl")
println("what")

roots = [
    "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/2b2t/2b2t15c10000e100dt",
    "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/2b2t/2b2t15c10000e_dt50_500_1000",
    "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/2b2t/2b2t15c1000e_dt10-",
    "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/2b2t/2b2t15c10000e_dt_10+",
    "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/2b2t/2b2t15c1000e_dt_10=-_int100-",
    "/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/2b2t15c10000e10+dt100-int",
    "/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/2b2t15c10000e10+dt90+int",
    "/project/theorie/n/N.Pfaffenzeller/results_project/2b2t/2b2t15c_20-dt100+int"
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/test_runs/10b10t15c500e_log_sweep"
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c1000e_lr5",
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c10000e50+500dt",
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c10000e100dt",
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c1000e_dt_low_between",
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c10000e_dt_mid_between",
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/big_runs/10b10t15c10000e_between_vals_high_dt",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/10b10t/10b10t15c1000e_10-dt80-int",
    #"/project/theorie/n/N.Pfaffenzeller/results_project/10b10t/10b10t15c1000e10-dt70+int"
    #"/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/results/test_runs/2b2t15c100e"
]

println("is")
results, failures = steffen_scores_all_runs_from_config(roots; thresholds=[0.8, 0.9])
results, failures = steffen_scores_all_runs_from_config_long(roots; thresholds=[0.8, 0.9])

println("good")
output_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/analysis_out"
prefix = "steffen_scores_2b2t_wout_og_$(Dates.format(now(), "yyyy-mm-dd_HHMM"))"
#prefix = "steffen_test"
paths = save_analysis_outputs(results, failures,
    output_dir;
    prefix=prefix
)

println("Saved: ", paths)
println("OK runs: ", nrow(results), " | failures: ", nrow(failures))
