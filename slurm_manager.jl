#!/usr/bin/env julia

"""
Slurm batch job manager for running parameter sweeps and replicate simulations.
Generates Slurm job scripts and manages job submission.
"""

using Pkg
Pkg.activate("./env_nethive_multiverse")
using ArgParse
using CSV
using DataFrames
using Flux
using JSON3
using MLDatasets
using Random
using Statistics
using Dates

include("run_simulation.jl")

output_dir_default = "slurm_results_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))"

function parse_slurm_args()
    s = ArgParseSettings(description = "Manage Slurm batch jobs for bee simulations")
    
    @add_arg_table! s begin
        "command"
            help = "Command to execute: generate, submit, status, or create-example"
            required = true
        "--config", "-c"
            help = "JSON configuration file for parameter sweep"
            arg_type = String
            required = true
        "--output-dir", "-o"
            help = "Output directory for results and job scripts"
            arg_type = String
            default = output_dir_default
        "--parent-folder", "-f"
            help = "Parent folder for all runs"
            arg_type = String
            default = "multi_task_simulations"
        "--job-name"
            help = "Base name for Slurm jobs"
            arg_type = String
            default = "bee_sim"
        "--partition", "-p"
            help = "Slurm partition to use"
            arg_type = String
            default = "th-ws"
        "--time", "-t"
            help = "Job time limit (HH:MM:SS)"
            arg_type = String
            default = "05:30:00"
        "--memory", "-m"
            help = "Memory per job (GB)"
            arg_type = Int
            default = 12
        "--cpus"
            help = "CPUs per job"
            arg_type = Int
            default = 4
        "--array-max"
            help = "Maximum number of simultaneous array jobs"
            arg_type = Int
            default = 100
        "--email"
            help = "Email for job notifications"
            arg_type = String
            default = ""
        "--dry-run"
            help = "Generate scripts but don't submit"
            action = :store_true
        "--replicates"
            help = "Number of replicates per parameter combination"
            arg_type = Int
            default = 1
        "--base-seed"
            help = "Base seed for reproducible replicate generation (auto-generated if not specified)"
            arg_type = Int
            default = nothing
        "--use-array"
            help = "Use job arrays instead of individual jobs"
            action = :store_true
    end
    
    return parse_args(s)
end

function load_parameter_sweep_config(config_path::String)
    """Load parameter sweep configuration from JSON file"""
    if !isfile(config_path)
        error("Parameter sweep configuration file not found: $config_path")
    end
    
    config_data = JSON3.read(read(config_path, String))
    # Convert JSON3.Object to Dict with string keys
    config_dict = Dict(String(k) => (v isa JSON3.Object ? Dict(String(kk) => vv for (kk, vv) in v) : v) for (k, v) in config_data)

    # Validate and update configuration for multi-task compatibility
    #return validate_and_update_config(config_dict)

    return config_dict
end

function validate_and_update_config(config::Dict)
    """Validate and update configuration to ensure multi-task compatibility"""
    base_config = config["base_config"]
    
    # Check if we have the new multi-task parameters
    has_new_params = haskey(base_config, "n_tasks") && haskey(base_config, "init_tasks_range")
    has_old_params = haskey(base_config, "init_n1_range") || haskey(base_config, "init_n2_range")
    
    if has_old_params && !has_new_params
        println("⚠️  Converting old single-task configuration to multi-task format...")
        
        # Convert old parameters to new multi-task format
        base_config["n_tasks"] = 2  # Assume 2 tasks for old configs
        
        # Convert init ranges
        if haskey(base_config, "init_n1_range") && haskey(base_config, "init_n2_range")
            # Use the larger range to be safe
            range1 = base_config["init_n1_range"]
            range2 = base_config["init_n2_range"] 
            init_range = [min(range1[1], range2[1]), max(range1[2], range2[2])]
            base_config["init_tasks_range"] = init_range
            
            # Remove old parameters
            delete!(base_config, "init_n1_range")
            delete!(base_config, "init_n2_range")
        else
            base_config["init_tasks_range"] = [1, 10]  # Default range
        end
        
        # Convert suppression probability
        if haskey(base_config, "init_q1_prob") || haskey(base_config, "init_q2_prob")
            q1_prob = get(base_config, "init_q1_prob", 0.0)
            q2_prob = get(base_config, "init_q2_prob", 0.0)
            base_config["init_suppression_prob"] = max(q1_prob, q2_prob)  # Use the higher probability
            
            # Remove old parameters
            delete!(base_config, "init_q1_prob")
            delete!(base_config, "init_q2_prob")
        else
            base_config["init_suppression_prob"] = 0.1  # Default probability
        end
        
        println("✅ Converted to multi-task format with $(base_config["n_tasks"]) tasks")
    end
    
    # Ensure required multi-task parameters are available (either in base_config or sweep_parameters)
    sweep_params = get(config, "sweep_parameters", Dict())
    
    # Check that critical parameters are specified somewhere
    critical_params = ["n_tasks"]
    for param in critical_params
        has_in_base = haskey(base_config, param)
        has_in_sweep = haskey(sweep_params, param)
        
        if !has_in_base && !has_in_sweep
            error("$param must be specified either in base_config or as a sweep parameter")
        end
        
        if has_in_base && has_in_sweep
            println("⚠️  Warning: $param appears in both base_config and sweep_parameters. Using sweep_parameters version.")
            delete!(base_config, param)  # Remove from base to avoid conflict
        end
    end
    
    # Ensure other required parameters have defaults if not specified
    default_params = [
        ("init_tasks_range", [1, 10]),
        ("init_suppression_prob", 0.1),
        ("interaction_kernel_prefactor_type", "standard_prefactor"),
        ("interaction_kernel_heaviside_arg_type", "difference")
    ]
    
    for (param, default_value) in default_params
        has_in_base = haskey(base_config, param)
        has_in_sweep = haskey(sweep_params, param)
        
        if !has_in_base && !has_in_sweep
            base_config[param] = default_value
            println("➕ Added missing parameter: $param = $default_value")
        end
    end
    
    return config
end

function generate_parameter_combinations(sweep_config::Dict)
    """Generate all parameter combinations from sweep configuration"""
    base_config = sweep_config["base_config"]
    sweep_params = sweep_config["sweep_parameters"]
    
    # Get parameter names and values
    param_names = collect(keys(sweep_params))
    param_values = [sweep_params[name] for name in param_names]
    
    # Generate all combinations
    combinations = []
    
    function generate_recursive(current_combo, remaining_params, remaining_values)
        if isempty(remaining_params)
            push!(combinations, copy(current_combo))
            return
        end
        
        param = remaining_params[1]
        values = remaining_values[1]
        
        for value in values
            current_combo[param] = value
            generate_recursive(current_combo, remaining_params[2:end], remaining_values[2:end])
        end
    end
    
    generate_recursive(copy(base_config), param_names, param_values)
    
    return combinations
end

function create_slurm_script(job_config::Dict, job_id::Int, output_dir::String)
    """Create a Slurm job script for a single parameter combination"""
    
    script_path = joinpath(output_dir, "scripts", "job_$(job_id).slurm")
    
    # Create scripts directory if it doesn't exist
    mkpath(dirname(script_path))
    
    open(script_path, "w") do io
        write(io, "#!/bin/bash\n")
        write(io, "#SBATCH --job-name=$(job_config["job_name"])_$(job_id)\n")
        write(io, "#SBATCH --partition=$(job_config["partition"])\n")
        write(io, "#SBATCH --time=$(job_config["time"])\n")
        write(io, "#SBATCH --mem=$(job_config["memory"])G\n")
        write(io, "#SBATCH --cpus-per-task=$(job_config["cpus"])\n")
        write(io, "#SBATCH --output=$(output_dir)/logs/job_$(job_id).out\n")
        write(io, "#SBATCH --error=$(output_dir)/logs/job_$(job_id).err\n")
        
        if !isempty(job_config["email"])
            write(io, "#SBATCH --mail-type=END,FAIL\n")
            write(io, "#SBATCH --mail-user=$(job_config["email"])\n")
        end
        
        write(io, "\n# Job information\n")
        write(io, "echo \"Job started at: \$(date)\"\n")
        write(io, "echo \"Job ID: \$SLURM_JOB_ID\"\n")
        write(io, "echo \"Node: \$SLURM_NODELIST\"\n")
        write(io, "echo \"Parameter combination: $(get(job_config, "param_id", "N/A"))\"\n")
        write(io, "echo \"Replicate: $(get(job_config, "replicate_id", "N/A"))\"\n")

        write(io, "\n# Load modules (required for cluster compatibility)\n")
        write(io, "echo \"Loading Julia module...\"\n")
        write(io, "module load julia\n")
        write(io, "echo \"Module loaded: \$(which julia)\"\n")

        write(io, "\n# Change to project directory\n")
        write(io, "echo \"Changing to project directory...\"\n")
        write(io, "cd \$SLURM_SUBMIT_DIR\n")
        write(io, "echo \"Current directory: \$(pwd)\"\n")
        
        write(io, "\n# Set Julia project environment\n")
        write(io, "export JULIA_PROJECT=./env_nethive_multiverse\n")

        write(io, "\n# Run simulation\n")
        
        # Generate config file path for this job
        config_path = joinpath(output_dir, "configs", "config_$(job_id).json")
        
        # Generate unique base name for this job
        param_id = get(job_config, "param_id", job_id)
        rep_id = get(job_config, "replicate_id", 1)
        job_base_name = "param_$(param_id)_rep_$(rep_id)"
        
        # Run simulation using the working pattern
        write(io, "echo \"Starting Julia simulation...\"\n")
        write(io, "julia run_simulation.jl --config $(config_path) --output-dir $(output_dir)/data --base-name $(job_base_name) --timestamp --verbose --save-results\n")
        
        write(io, "\necho \"Job finished at: \$(date)\"\n")
    end
    
    return script_path
end

function create_submission_script(output_dir::String, script_paths::Vector{String})
    """Create a master script to submit all jobs (individual mode)"""
    submit_script = joinpath(output_dir, "submit_all.sh")
    
    open(submit_script, "w") do io
        write(io, "#!/bin/bash\n")
        write(io, "# Master submission script for bee simulation parameter sweep\n\n")
        
        write(io, "echo \"Submitting $(length(script_paths)) jobs...\"\n\n")
        
        for (i, script_path) in enumerate(script_paths)
            write(io, "echo \"Submitting job $i: $(basename(script_path))\"\n")
            write(io, "sbatch $(script_path)\n")
            write(io, "sleep 0.1  # Small delay to avoid overwhelming scheduler\n\n")
        end
        
        write(io, "echo \"All jobs submitted!\"\n")
        write(io, "echo \"Check status with: squeue -u \$USER\"\n")
        write(io, "echo \"Monitor progress in: $(output_dir)/logs/\"\n")
    end
    
    # Make executable
    run(`chmod +x $(submit_script)`)
    
    return submit_script
end

function create_job_array_script(output_dir::String, configs::Vector{Dict}, args::Dict)
    """Create a single job array script instead of individual scripts"""
    
    # Create the main job array script
    array_script = joinpath(output_dir, "job_array.slurm")
    
    # Calculate array size
    n_jobs = length(configs)
    max_concurrent = min(args["array-max"], n_jobs)
    
    open(array_script, "w") do io
        write(io, "#!/bin/bash\n")
        write(io, "#SBATCH --job-name=bee_sim_array\n")
        write(io, "#SBATCH --partition=$(args["partition"])\n") 
        write(io, "#SBATCH --time=$(args["time"])\n")
        write(io, "#SBATCH --mem=$(args["memory"])G\n")
        write(io, "#SBATCH --cpus-per-task=$(args["cpus"])\n")
        write(io, "#SBATCH --array=1-$(n_jobs)%$(max_concurrent)\n")  # Job array with throttling
        write(io, "#SBATCH --output=$(output_dir)/logs/job_%A_%a.out\n")  # %A = job ID, %a = array index
        write(io, "#SBATCH --error=$(output_dir)/logs/job_%A_%a.err\n")
        
        if !isempty(args["email"])
            write(io, "#SBATCH --mail-type=END,FAIL\n")
            write(io, "#SBATCH --mail-user=$(args["email"])\n")
        end
        
        write(io, "\n# Job array information\n")
        write(io, "echo \"Job array started at: \$(date)\"\n")
        write(io, "echo \"Job ID: \$SLURM_JOB_ID\"\n") 
        write(io, "echo \"Array Job ID: \$SLURM_ARRAY_JOB_ID\"\n")
        write(io, "echo \"Array Task ID: \$SLURM_ARRAY_TASK_ID\"\n")
        write(io, "echo \"Node: \$SLURM_NODELIST\"\n")
        
        write(io, "\n# Load modules (required for cluster compatibility)\n")
        write(io, "echo \"Loading Julia module...\"\n")
        write(io, "module load julia\n")
        write(io, "echo \"Module loaded: \$(which julia)\"\n")
        
        write(io, "\n# Change to project directory\n")
        write(io, "echo \"Changing to project directory...\"\n")
        write(io, "cd \$SLURM_SUBMIT_DIR\n")
        write(io, "echo \"Current directory: \$(pwd)\"\n")
        
        write(io, "\n# Set Julia project environment\n")
        write(io, "export JULIA_PROJECT=./env_nethive_multiverse\n")
        
        write(io, "\n# Configuration mapping based on array task ID\n")
        write(io, "case \$SLURM_ARRAY_TASK_ID in\n")
        
        for (i, config) in enumerate(configs)
            param_id = get(config, "param_id", i)
            rep_id = get(config, "replicate_id", 1)
            job_base_name = "param_$(param_id)_rep_$(rep_id)"
            config_path = joinpath(output_dir, "configs", "config_$(i).json")
            
            write(io, "  $(i))\n")
            write(io, "    echo \"Running parameter combination $(param_id), replicate $(rep_id)\"\n")
            write(io, "    CONFIG_FILE=\"$(config_path)\"\n")
            write(io, "    BASE_NAME=\"$(job_base_name)\"\n")
            write(io, "    ;;\n")
        end
        
        write(io, "  *)\n")
        write(io, "    echo \"Error: Unknown array task ID \$SLURM_ARRAY_TASK_ID\"\n")
        write(io, "    exit 1\n")
        write(io, "    ;;\n")
        write(io, "esac\n")
        
        write(io, "\n# Run simulation\n")
        write(io, "echo \"Starting Julia simulation...\"\n")
        write(io, "julia run_simulation.jl --config \$CONFIG_FILE --output-dir $(output_dir)/data --base-name \$BASE_NAME --timestamp --verbose --save-results\n")
        
        write(io, "\necho \"Job finished at: \$(date)\"\n")
    end
    
    # Make executable
    run(`chmod +x $(array_script)`)
    
    return array_script
end

function create_array_submission_script(output_dir::String, array_script::String)
    """Create submission script for job array"""
    submit_script = joinpath(output_dir, "submit_array.sh")
    
    open(submit_script, "w") do io
        write(io, "#!/bin/bash\n")
        write(io, "# Submit job array for bee simulation parameter sweep\n\n")
        
        write(io, "echo \"Submitting job array: $(basename(array_script))\"\n")
        write(io, "sbatch $(array_script)\n\n")
        
        write(io, "echo \"Job array submitted!\"\n")
        write(io, "echo \"Check status with: squeue -u \$USER\"\n")
        write(io, "echo \"Monitor progress in: $(output_dir)/logs/\"\n")
    end
    
    # Make executable
    run(`chmod +x $(submit_script)`)
    
    return submit_script
end

function generate_jobs(sweep_config::Dict, args::Dict)
    """Generate all Slurm job scripts for parameter sweep"""
    output_dir = joinpath(args["parent-folder"], args["output-dir"])
    use_array = args["use-array"]
    
    # Create directory structure
    mkpath(joinpath(output_dir, "scripts"))
    mkpath(joinpath(output_dir, "configs"))
    mkpath(joinpath(output_dir, "logs"))
    mkpath(joinpath(output_dir, "data"))
    
    # Generate parameter combinations
    combinations = generate_parameter_combinations(sweep_config)
    
    println("Generated $(length(combinations)) parameter combinations")
    
    # Create job configurations with seeds
    all_configs = Dict[]
    
    # Handle base seed - generate random one if not specified
    base_seed = if args["base-seed"] !== nothing
        args["base-seed"]
    else
        generated_seed = abs(hash(time_ns())) % Int32
        println("🎲 Generated random base seed: $generated_seed")
        println("   To reproduce this exact sweep, use: --base-seed $generated_seed")
        generated_seed
    end
    
    job_counter = 1
    
    for (param_id, combo) in enumerate(combinations)
        # Create separate configs for each replicate
        for rep in 1:args["replicates"]
            # Create unique seed for this parameter combination and replicate
            unique_seed = base_seed + (param_id - 1) * 1000 + rep
            
            # Add seed to configuration
            combo_with_seed = copy(combo)
            combo_with_seed["seed"] = unique_seed
            combo_with_seed["param_id"] = param_id
            combo_with_seed["replicate_id"] = rep
            
            # Save configuration file with unique name
            config_path = joinpath(output_dir, "configs", "config_$(job_counter).json")
            open(config_path, "w") do io
                JSON3.pretty(io, combo_with_seed)
            end
            
            push!(all_configs, combo_with_seed)
            
            println("Created config $job_counter: param_combo_$(param_id)_rep_$(rep) (seed: $unique_seed)")
            job_counter += 1
        end
    end
    
    n_param_combinations = length(combinations)
    n_replicates = args["replicates"]
    total_jobs = length(all_configs)
    
    if use_array
        # Create job array
        println("\n🚀 Using JOB ARRAY mode:")
        
        array_script = create_job_array_script(output_dir, all_configs, args)
        submit_script = create_array_submission_script(output_dir, array_script)
        
        println("Generated 1 job array script with $total_jobs tasks:")
        println("  $n_param_combinations parameter combinations")
        println("  $n_replicates replicates each") 
        println("  Base seed: $(args["base-seed"])")
        println("  Max concurrent: $(min(args["array-max"], total_jobs))")
        println("Configuration files: $(output_dir)/configs/")
        println("Job array script: $(array_script)")
        println("Submission script: $(submit_script)")
        println("\nTo submit job array, run: $(submit_script)")
        
        return [array_script], submit_script
        
    else
        # Create individual job scripts (original mode)
        println("\n📝 Using INDIVIDUAL JOBS mode:")
        
        script_paths = String[]
        
        for (i, config) in enumerate(all_configs)
            job_config = Dict(
                "job_name" => args["job-name"],
                "partition" => args["partition"],
                "time" => args["time"],
                "memory" => args["memory"],
                "cpus" => args["cpus"],
                "array_max" => args["array-max"],
                "email" => args["email"],
                "replicates" => 1,  # Each job is a single replicate
                "param_id" => config["param_id"],
                "replicate_id" => config["replicate_id"]
            )
            
            script_path = create_slurm_script(job_config, i, output_dir)
            push!(script_paths, script_path)
        end
        
        submit_script = create_submission_script(output_dir, script_paths)
        
        println("Generated $total_jobs individual job scripts:")
        println("  $n_param_combinations parameter combinations")
        println("  $n_replicates replicates each") 
        println("  Base seed: $base_seed")
        println("Configuration files: $(output_dir)/configs/")
        println("Job scripts: $(output_dir)/scripts/")
        println("Submission script: $(submit_script)")
        println("\nTo submit all jobs, run: $(submit_script)")
        
        return script_paths, submit_script
    end
end

function submit_jobs(submit_script::String, dry_run::Bool=false)
    """Submit all jobs to Slurm"""
    if dry_run
        println("DRY RUN: Would execute: $(submit_script)")
        return
    end
    
    println("Submitting jobs...")
    run(`$(submit_script)`)
    println("Jobs submitted!")
end

function main()
    """Main function for Slurm job management"""
    args = parse_slurm_args()
    
    command = args["command"]
    
    if !isdir(args["parent-folder"])
        mkpath(args["parent-folder"])
    end

    if command == "generate"
        sweep_config = load_parameter_sweep_config(args["config"])
        script_paths, submit_script = generate_jobs(sweep_config, args)
        
        if !args["dry-run"]
            println("\nReady to submit! Run:")
            println("  julia slurm_manager.jl submit --config $(args["config"]) --output-dir $(args["output-dir"])")
            println("Or directly:")
            println("  $(submit_script)")
        end
    
    elseif command == "create-example"
        example_path = joinpath(args["parent-folder"], "example_multi_task_sweep.json")
        create_example_config(example_path)
        
    elseif command == "submit"
        submit_script = joinpath(args["parent-folder"], args["output-dir"], "submit_all.sh")
        
        if !isfile(submit_script)
            error("Submit script not found. Run 'generate' command first.")
        end
        
        submit_jobs(submit_script, args["dry-run"])
        
    elseif command == "status"
        output_dir = joinpath(args["parent-folder"], args["output-dir"])
        check_job_status(output_dir)
        
    else
        error("Unknown command: $command. Use: generate, create-example, submit, or status")
    end
end


function check_job_status(output_dir::String)
    """Check status of submitted jobs"""
    println("Checking job status...")
    
    # Show current queue status
    try
        run(`squeue -u $(ENV["USER"])`)
    catch
        println("Could not run squeue - check if you're on a Slurm system")
    end
    
    # Check for completed jobs
    data_dir = joinpath(output_dir, "data")
    if isdir(data_dir)
        # Count unique simulations by looking for config.json files
        config_files = filter(f -> f == "config.json", 
                            vcat([readdir(joinpath(data_dir, d), join=false) for d in readdir(data_dir) 
                                 if isdir(joinpath(data_dir, d))]...))
        println("\nCompleted simulations: $(length(config_files))")
        
        # Also count CSV files for reference  
        csv_files = []
        for subdir in readdir(data_dir)
            subdir_path = joinpath(data_dir, subdir)
            if isdir(subdir_path)
                append!(csv_files, filter(f -> endswith(f, ".csv"), readdir(subdir_path)))
            end
        end
        println("Total CSV files: $(length(csv_files))")
    end
end

function create_example_config(output_path::String)
    """Create an example multi-task parameter sweep configuration"""
    
    example_config = Dict(
        "base_config" => Dict(
            "n_bees" => 5,
            "dataset_names" => ["mnist", "fashion_mnist"],
            "n_epochs" => 10,
            "n_steps_per_epoch" => 100,
            "production_rate" => 5.0,
            "interaction_rate" => 10.0,
            "learning_rate" => 0.001,
            "punish_rate" => 0.0,
            "lambda_sensitivity" => 10.0,
            "batch_size" => 32,
            "save_nn_epochs" => 0,
            "seed" => nothing
        ),
        "sweep_parameters" => Dict(
            "lambda_sensitivity" => [1.0, 10.0, 100.0],
            "interaction_rate" => [5.0, 10.0, 20.0],
            "production_rate" => [100.0, 500.0, 1000.0]
        )
    )
    
    open(output_path, "w") do io
        JSON3.pretty(io, example_config)
    end
    
    println("✅ Created example multi-task parameter sweep configuration:")
    println("   File: $output_path")
    println("   Parameter combinations: $(length(example_config["sweep_parameters"]["lambda_sensitivity"]) * 
                                        length(example_config["sweep_parameters"]["interaction_rate"]) * 
                                        length(example_config["sweep_parameters"]["production_rate"]))")
    println("\nExample usage:")
    println("   julia slurm_manager.jl generate --config $output_path")
end

# Run main if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


