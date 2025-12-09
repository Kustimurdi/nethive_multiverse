#register_mod_add(target_input, )

struct TaskConfig
    n_classes::Int
    features_dimension::Int
    n_per_class_train::Int
    n_per_class_test::Int
    sampling_gauss_sigma::Float64
    use_per_class_variance::Bool
    variance_bounds::Tuple{Float64, Float64}
    center_generation_bounds::Tuple{Float64, Float64}
    n_tasks::Int

    function TaskConfig(n_classes::Int,
                        n_tasks::Int,
                        features_dimension::Int,
                        n_per_class_train::Int,
                        n_per_class_test::Int,
                        use_per_class_variance::Bool,
                        variance_bounds::Tuple{Float64, Float64},
                        center_generation_bounds::Tuple{Float64, Float64};
                        sampling_gauss_sigma::Float64=1.0)
        
        # Validate parameters
        if n_classes < 2
            throw(ArgumentError("Number of classes must be at least 2"))
        end
        if features_dimension < 1
            throw(ArgumentError("Features dimension must be at least 1"))
        end
        if n_per_class_train < 1
            throw(ArgumentError("Number of training samples per class must be at least 1"))
        end
        if n_per_class_test < 1
            throw(ArgumentError("Number of test samples per class must be at least 1"))
        end
        if sampling_gauss_sigma <= 0
            throw(ArgumentError("Sampling Gaussian sigma must be positive"))
        end
        if variance_bounds[1] < 0 || variance_bounds[2] <= variance_bounds[1]
            throw(ArgumentError("Invalid variance bounds: $variance_bounds"))
        end
        if center_generation_bounds[1] >= center_generation_bounds[2]
            throw(ArgumentError("Invalid center generation bounds: $center_generation_bounds"))
        end
        if n_tasks < 1
            throw(ArgumentError("Number of tasks must be at least 1"))
        end
        
        return new(n_classes,
                features_dimension,
                n_per_class_train,
                n_per_class_test,
                sampling_gauss_sigma,
                use_per_class_variance,
                variance_bounds,
                center_generation_bounds,
                n_tasks)
    end
end


function prepare_all_gauss_loaders(config::TaskConfig; batchsize::Int=32, shuffle_train::Bool=true)
    dataset = create_dataset(config)
    all_datasets = generate_rotated_tasks(dataset, config.n_tasks)
    loaders = Dict{Symbol, Dict{String, Any}}()
    for (i, task) in enumerate(all_datasets)
        println("Preparing loaders for task $(i)...")
        train_loader, test_loader = create_gauss_loaders(
            task.train_data, task.test_data; 
            batchsize=batchsize, 
            shuffle_train=shuffle_train
        )
        loaders[Symbol("task_$(i)")] = Dict(
            "train" => train_loader,
            "test" => test_loader,
            "rotation" => task.rotation
        )
    end
    return loaders
end

function create_gauss_model_template(config::TaskConfig)
    function model_template()
        model = Chain(
            Dense(config.features_dimension, 128, relu),
            Dense(128, 64, relu),
            Dense(64, config.n_classes)
        )
        return model
    end
    @info "Created gauss model template" input_dim=config.features_dimension output_dim=config.n_classes
    return model_template
end


function create_gauss_loaders(train_data::Tuple, test_data::Tuple; batchsize=32, shuffle_train::Bool=true)
    train_x, train_y = train_data
    test_x, test_y = test_data

    train_y_onehot = Flux.onehotbatch(train_y, 1:maximum(train_y))
    test_y_onehot = Flux.onehotbatch(test_y, 1:maximum(test_y))
    
    # Create proper Flux DataLoaders
    train_loader = Flux.DataLoader(
        (train_x, train_y_onehot), 
        batchsize=batchsize, 
        shuffle=shuffle_train
    )
    
    test_loader = Flux.DataLoader(
        (test_x, test_y_onehot), 
        batchsize=batchsize, 
        shuffle=false  # Don't shuffle test data
    )
    
    return train_loader, test_loader

end


function create_dataset(config::TaskConfig)
    # Generate centers based on configuration
    centers = generate_random_centers(
        config.n_classes, config.features_dimension, 
        config.center_generation_bounds
        )
    #centers = generate_orthogonal_centers(config.n_classes, config.features_dimension, config.class_center_radius)
    
    # Generate data based on variance configuration
    if config.use_per_class_variance
        # Generate per-class variances
        class_variances = generate_per_class_variances(config.n_classes, config.variance_bounds)
        train_features, train_labels = generate_gaussian_classification_data(centers, class_variances, config.n_per_class_train)
        test_features, test_labels = generate_gaussian_classification_data(centers, class_variances, config.n_per_class_test)
    else
        # Use uniform variance across all classes
        train_features, train_labels = generate_gaussian_classification_data(centers, config.sampling_gauss_sigma, config.n_per_class_train)
        test_features, test_labels = generate_gaussian_classification_data(centers, config.sampling_gauss_sigma, config.n_per_class_test)
    end
    
    #train_labels = Flux.onehotbatch(train_labels, 1:config.n_classes)
    #test_labels = Flux.onehotbatch(test_labels, 1:config.n_classes)
    #return Flux.DataLoader((train_features', train_labels), batchsize=batchsize, shuffle=true),
            #Flux.DataLoader((test_features', test_labels), batchsize=batchsize, shuffle=true),
            #(train_features', train_labels),
            #(test_features', test_labels)
    return (
        train_data = (train_features', train_labels), 
        test_data = (test_features', test_labels),
        n_classes = config.n_classes
        )
end

function generate_rotated_tasks(dataset::NamedTuple, n_tasks::Int)
    train_x, train_y = dataset.train_data
    test_x, test_y = dataset.test_data

    labels = unique(train_y)
    n_classes = length(labels)

    tasks = NamedTuple[]

    for task_id in 0:n_tasks-1
        rot_train_labels = mod1.(train_y .+ task_id, n_classes)
        rot_test_labels = mod1.(test_y .+ task_id, n_classes)
        push!(tasks, (
            train_data = (train_x, rot_train_labels),
            test_data = (test_x, rot_test_labels),
            rotation = task_id
        ))
        println("data for task $(task_id):")
        println("  Train dataset size (length of rot_train_labels): $(length(rot_train_labels))")
        println("  Test dataset size: $(length(rot_test_labels))")
        println("train labels min=", minimum(rot_train_labels), " max=", maximum(rot_train_labels))
        println("test  labels min=", minimum(rot_test_labels), " max=", maximum(rot_test_labels))
    end

    return tasks

end


"""
    generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigma::Float64, n_per_class::Int)

Generates synthetic classification data with uniform variance across all classes.

- `centers`: a vector of center vectors, one per class
- `sigma`: standard deviation for Gaussian sampling
- `n_per_class`: number of samples per class

Returns:
- `X`: matrix of features (n_total × d)
- `y`: vector of labels (Int) (n_total)
"""
function generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigma::Float64, n_per_class::Int)
    d = length(centers[1])
    k = length(centers)  # number of classes

    X = Matrix{Float64}(undef, 0, d)
    y = Int[]

    for (label, center) in enumerate(centers)
        cov = (sigma^2) * I
        dist = MvNormal(center, cov)
        samples = rand(dist, n_per_class)'  # n_per_class × d
        X = vcat(X, samples)
        append!(y, fill(label, n_per_class))
    end

    return X, y
end

"""
    generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigmas::Vector{Float64}, n_per_class::Int)

Generates synthetic classification data with per-class variance.

- `centers`: a vector of center vectors, one per class
- `sigmas`: a vector of standard deviations, one per class
- `n_per_class`: number of samples per class

Returns:
- `X`: matrix of features (n_total × d)
- `y`: vector of labels (Int) (n_total)
"""
function generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigmas::Vector{Float64}, n_per_class::Int)
    d = length(centers[1])
    k = length(centers)  # number of classes
    
    @assert length(sigmas) == k "Number of sigmas must match number of classes"

    X = Matrix{Float64}(undef, 0, d)
    y = Int[]

    for (label, (center, sigma)) in enumerate(zip(centers, sigmas))
        cov = (sigma^2) * I
        dist = MvNormal(center, cov)
        samples = rand(dist, n_per_class)'  # n_per_class × d
        X = vcat(X, samples)
        append!(y, fill(label, n_per_class))
    end

    return X, y
end

"""
    generate_orthogonal_centers(n_classes::Int, d_features::Int; radius=5.0)

Returns a vector of class centers placed on orthogonal axes.
"""
function generate_orthogonal_centers(n_classes::Int, d_features::Int, radius)
    @assert n_classes <= d_features "You need at least as many dimensions as classes"

    return [radius * unit_vector(i, d_features) for i in 1:n_classes]
end

# Helper: returns a unit vector with 1.0 at index `i`
unit_vector(i, dim) = [j == i ? 1.0 : 0.0 for j in 1:dim]

"""
    generate_random_centers(n_classes::Int, d_features::Int, bounds::Tuple{Float64, Float64})

Returns a vector of randomly generated class centers within the specified bounds.
"""
function generate_random_centers(n_classes::Int, d_features::Int, bounds::Tuple{Float64, Float64})
    min_bound, max_bound = bounds
    centers = Vector{Vector{Float64}}()
    
    for i in 1:n_classes
        center = min_bound .+ (max_bound - min_bound) .* rand(d_features)
        push!(centers, center)
    end
    
    return centers
end

"""
    generate_per_class_variances(n_classes::Int, variance_bounds::Tuple{Float64, Float64})

Returns a vector of random variances, one per class, within the specified bounds.
"""
function generate_per_class_variances(n_classes::Int, variance_bounds::Tuple{Float64, Float64})
    min_var, max_var = variance_bounds
    return min_var .+ (max_var - min_var) .* rand(n_classes)
end


function create_loaders(dataset; batch_size::Int=32, shuffle_train::Bool=true)
    train_x, train_y = dataset.train_data
    test_x, test_y = dataset.test_data
    
    # Create proper Flux DataLoaders
    train_loader = Flux.DataLoader(
        (train_x, train_y), 
        batchsize=batch_size, 
        shuffle=shuffle_train
    )
    
    test_loader = Flux.DataLoader(
        (test_x, test_y), 
        batchsize=batch_size, 
        shuffle=false  # Don't shuffle test data
    )
    
    return train_loader, test_loader
end








# --- prob useless ---
function create_linear_dataset(setsize::Int)
    # Simple linear regression data (y = mx + b)
    x = rand(setsize) * 10
    m, b = 2.0, 5.0  # slope and intercept
    y = m * x .+ b + randn(setsize) * 0.5  # Adding noise
    return reshape(Float32.(x), 1, :), reshape(Float32.(y), 1, :)
end

function create_sin_dataset(n_peaks, which_peak, setsize::Int)
    features = rand(setsize) * pi *n_peaks |> x -> reshape(x, 1, :)
    temp = deepcopy(features)
    temp[(temp .< (which_peak - 1)*pi) .| (temp .> which_peak*pi)] .= 0
    labels = abs.(sin.(temp)) * 10
    return Float32.(features), Float32.(labels)
end

function sample_gaussian_around(x::Vector, sigma::Float64, n::Int)
    d = length(x)
    cov = (sigma^2) * I
    dist = MvNormal(x, cov)
    return rand(dist, n)
end

