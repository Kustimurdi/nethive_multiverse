function register_cifar10!(target_input_dim::Int=3072, target_output_dim::Int=10)
    if haskey(DATASET_REGISTRY, :cifar10)
        @info "CIFAR-10 already registered, skipping"
        return DATASET_REGISTRY[:cifar10]
    end
    
    @info "Registering CIFAR-10 dataset..."
    
    # Load CIFAR-10 data
    train_data = MLDatasets.CIFAR10(:train)
    test_data = MLDatasets.CIFAR10(:test)
    
    # Convert to Float32 and flatten images (32×32×3 = 3072)
    train_x = reshape(Float32.(train_data.features), 3072, :)
    test_x = reshape(Float32.(test_data.features), 3072, :)
    
    # Convert labels to one-hot encoding (classes 0-9)
    train_y = Flux.onehotbatch(train_data.targets, 0:9)
    test_y = Flux.onehotbatch(test_data.targets, 0:9)
    
    # Apply padding if needed
    if target_input_dim > 3072
        padding_size = target_input_dim - 3072
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end
    
    if target_output_dim > 10
        padding_size = target_output_dim - 10
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end

    dataset = TaskDataset(
        :cifar10,
        (32, 32, 3),
        10,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:cifar10] = dataset
    @info "CIFAR-10 registered successfully" input_shape=(32, 32, 3) n_classes=10 padded_dim=target_input_dim
    
    return dataset
end

function register_air_quality!(target_input)