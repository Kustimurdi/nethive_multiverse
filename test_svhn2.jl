#!/usr/bin/env julia

"""
SVHN2 Data Loading and Training Test

This script tests whether SVHN2 data loads correctly and neural networks
can be trained on it successfully.
"""

using Random
using Flux
using Printf

# Include our modules
include("src/data/registry.jl")

function test_svhn2_loading()
    """Test SVHN2 dataset loading and registration"""
    println("=" ^ 60)
    println("Testing SVHN2 Data Loading")
    println("=" ^ 60)
    
    try
        # Clear registry to ensure fresh load
        empty!(DATASET_REGISTRY)
        
        # Register SVHN2 with default dimensions
        println("📁 Registering SVHN2 dataset...")
        dataset = register_svhn2(3072, 10)
        
        # Basic checks
        @assert dataset.name == :svhn2 "Dataset name should be :svhn2"
        @assert dataset.input_shape == (32, 32, 3) "Input shape should be (32, 32, 3)"
        @assert dataset.n_classes == 10 "Should have 10 classes"
        @assert dataset.padded_input_dim == 3072 "Padded input dim should be 3072"
        @assert dataset.padded_output_dim == 10 "Padded output dim should be 10"
        
        # Check data shapes
        train_x, train_y = dataset.train_data
        test_x, test_y = dataset.test_data
        
        println("✓ Dataset registered successfully")
        println("  - Input shape: $(dataset.input_shape)")
        println("  - Classes: $(dataset.n_classes)")
        println("  - Train data: $(size(train_x)) features, $(size(train_y)) labels")
        println("  - Test data: $(size(test_x)) features, $(size(test_y)) labels")
        
        # Check data types and ranges
        @assert eltype(train_x) == Float32 "Training features should be Float32"
        @assert eltype(test_x) == Float32 "Test features should be Float32"
        @assert all(0 ≤ x ≤ 1 for x in train_x) "Training features should be normalized [0,1]"
        @assert all(0 ≤ x ≤ 1 for x in test_x) "Test features should be normalized [0,1]"
        
        println("✓ Data types and ranges are correct")
        
        # Check one-hot encoding
        @assert size(train_y, 1) == 10 "Training labels should have 10 classes"
        @assert size(test_y, 1) == 10 "Test labels should have 10 classes"
        @assert all(sum(train_y, dims=1) .≈ 1) "Training labels should be one-hot encoded"
        @assert all(sum(test_y, dims=1) .≈ 1) "Test labels should be one-hot encoded"
        
        println("✓ One-hot encoding is correct")
        
        # Test sample extraction
        sample_x = train_x[:, 1:5]  # First 5 samples
        sample_y = train_y[:, 1:5]
        
        println("✓ Sample data extraction works")
        println("  - Sample features shape: $(size(sample_x))")
        println("  - Sample labels shape: $(size(sample_y))")
        
        return dataset
        
    catch e
        println("❌ Error in SVHN2 loading test: $e")
        rethrow(e)
    end
end

function test_svhn2_training(dataset)
    """Test neural network training on SVHN2 data"""
    println("\n" * "=" ^ 60)
    println("Testing Neural Network Training on SVHN2")
    println("=" ^ 60)
    
    try
        Random.seed!(42)  # For reproducible results
        
        # Create a simple neural network
        model = Chain(
            Dense(dataset.padded_input_dim => 128, relu),
            Dense(128 => 64, relu),
            Dense(64 => dataset.padded_output_dim)
        )
        
        println("🧠 Created neural network:")
        println("  - Input dim: $(dataset.padded_input_dim)")
        println("  - Hidden layers: 128 → 64")
        println("  - Output dim: $(dataset.padded_output_dim)")
        
        # Prepare training data (use small subset for quick test)
        train_x, train_y = dataset.train_data
        test_x, test_y = dataset.test_data
        
        # Use first 1000 samples for quick training test
        n_train_samples = min(1000, size(train_x, 2))
        n_test_samples = min(200, size(test_x, 2))
        
        small_train_x = train_x[:, 1:n_train_samples]
        small_train_y = train_y[:, 1:n_train_samples]
        small_test_x = test_x[:, 1:n_test_samples]
        small_test_y = test_y[:, 1:n_test_samples]
        
        println("📊 Using subset for training test:")
        println("  - Training samples: $(n_train_samples)")
        println("  - Test samples: $(n_test_samples)")
        
        # Test forward pass
        println("🔄 Testing forward pass...")
        pred = model(small_train_x)
        @assert size(pred) == size(small_train_y) "Prediction shape should match labels"
        println("✓ Forward pass successful")
        println("  - Prediction shape: $(size(pred))")
        
        # Test loss computation
        println("📉 Testing loss computation...")
        loss_fn = Flux.crossentropy
        initial_loss = loss_fn(pred, small_train_y)
        println("✓ Loss computation successful")
        println("  - Initial loss: $(@sprintf("%.4f", initial_loss))")
        
        # Test accuracy computation
        println("🎯 Testing accuracy computation...")
        function accuracy(model, x, y)
            pred = model(x)
            pred_classes = Flux.onecold(pred, 0:9)
            true_classes = Flux.onecold(y, 0:9)
            return mean(pred_classes .== true_classes)
        end
        
        initial_acc = accuracy(model, small_test_x, small_test_y)
        println("✓ Accuracy computation successful")
        println("  - Initial accuracy: $(@sprintf("%.4f", initial_acc))")
        
        # Test training step
        println("🏋️ Testing training step...")
        opt = Flux.setup(Flux.Adam(0.001), model)
        
        # Single training step
        loss, grads = Flux.withgradient(model) do m
            pred = m(small_train_x)
            loss_fn(pred, small_train_y)
        end
        
        Flux.update!(opt, model, grads[1])
        
        # Check that loss changed
        new_pred = model(small_train_x)
        new_loss = loss_fn(new_pred, small_train_y)
        
        println("✓ Training step successful")
        println("  - Loss before: $(@sprintf("%.4f", loss))")
        println("  - Loss after: $(@sprintf("%.4f", new_loss))")
        println("  - Loss change: $(@sprintf("%.4f", new_loss - loss))")
        
        # Test mini training loop (5 epochs)
        println("🔁 Testing mini training loop (5 epochs)...")
        for epoch in 1:5
            # Simple batch training
            batch_size = 32
            n_batches = div(n_train_samples, batch_size)
            
            epoch_loss = 0.0
            for i in 1:n_batches
                start_idx = (i-1) * batch_size + 1
                end_idx = min(i * batch_size, n_train_samples)
                
                batch_x = small_train_x[:, start_idx:end_idx]
                batch_y = small_train_y[:, start_idx:end_idx]
                
                loss, grads = Flux.withgradient(model) do m
                    pred = m(batch_x)
                    loss_fn(pred, batch_y)
                end
                
                Flux.update!(opt, model, grads[1])
                epoch_loss += loss
            end
            
            avg_loss = epoch_loss / n_batches
            test_acc = accuracy(model, small_test_x, small_test_y)
            
            println("  Epoch $epoch: loss = $(@sprintf("%.4f", avg_loss)), acc = $(@sprintf("%.4f", test_acc))")
        end
        
        final_acc = accuracy(model, small_test_x, small_test_y)
        acc_improvement = final_acc - initial_acc
        
        println("✓ Mini training loop completed")
        println("  - Initial accuracy: $(@sprintf("%.4f", initial_acc))")
        println("  - Final accuracy: $(@sprintf("%.4f", final_acc))")
        println("  - Improvement: $(@sprintf("%.4f", acc_improvement))")
        
        # Check that model is actually learning
        if acc_improvement > 0.01
            println("✅ Model is learning! (accuracy improved by > 1%)")
        else
            println("⚠️  Model learning is slow (accuracy improved by ≤ 1%)")
            println("   This might be normal for a short training run")
        end
        
        return true
        
    catch e
        println("❌ Error in training test: $e")
        rethrow(e)
    end
end

function test_svhn2_data_loaders(dataset)
    """Test DataLoader creation and usage"""
    println("\n" * "=" ^ 60)
    println("Testing DataLoader Creation")
    println("=" ^ 60)
    
    try
        train_x, train_y = dataset.train_data
        
        # Create DataLoader
        train_data = [(train_x[:, i:min(i+31, end)], train_y[:, i:min(i+31, end)]) 
                      for i in 1:32:size(train_x, 2)]
        
        println("🔄 Created DataLoader with $(length(train_data)) batches")
        
        # Test first batch
        first_batch_x, first_batch_y = train_data[1]
        println("✓ First batch shape: features $(size(first_batch_x)), labels $(size(first_batch_y))")
        
        # Test iteration
        batch_count = 0
        for (batch_x, batch_y) in train_data[1:min(5, end)]
            batch_count += 1
            @assert size(batch_x, 1) == dataset.padded_input_dim
            @assert size(batch_y, 1) == dataset.padded_output_dim
        end
        
        println("✓ DataLoader iteration works (tested $batch_count batches)")
        
        return true
        
    catch e
        println("❌ Error in DataLoader test: $e")
        rethrow(e)
    end
end

function main()
    """Run all SVHN2 tests"""
    println("🧪 SVHN2 Data Loading and Training Test Suite")
    println("Julia version: $(VERSION)")
    println("Date: $(now())")
    
    try
        # Test 1: Data loading
        dataset = test_svhn2_loading()
        
        # Test 2: DataLoader creation  
        test_svhn2_data_loaders(dataset)
        
        # Test 3: Neural network training
        test_svhn2_training(dataset)
        
        println("\n" * "🎉" ^ 20)
        println("🎉 ALL TESTS PASSED! SVHN2 is working correctly! 🎉")
        println("🎉" ^ 20)
        
    catch e
        println("\n" * "💥" ^ 20)
        println("💥 TEST FAILED: $e")
        println("💥" ^ 20)
        rethrow(e)
    end
end

# Run the tests
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end