function load_gauss_dataset(file_path::String)
    @load file_path all_datasets
    return all_datasets
end

function load_gauss_metadata(file_dir::String)
    config_path = joinpath(file_dir, "config.json")
    if isfile(config_path)
        config = JSON3.read(open(config_path), Dict{String,Any})
        return config
    else
        error("Metadata file not found at: $config_path")
    end
end

function prepare_gaussset(all_datasets, rotation_index::Int, batchsize, shuffle::Bool=true)
    dataset = nothing
    #dataset = all_datasets[task_index]
    for ds in all_datasets
        if ds.rotation == rotation_index
            dataset = ds
            break
        end
    end
    if dataset === nothing
        error("Dataset with rotation_index $rotation_index not found.")
    end
    
    # Prepare DataLoaders
    train_loader, test_loader = create_gauss_loaders(dataset.train_data, dataset.test_data; batchsize=batchsize, shuffle_train=shuffle)

    return train_loader, test_loader
end