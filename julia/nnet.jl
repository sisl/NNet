module SimpleNNet

export NNet, evaluate_network, evaluate_network_multiple, num_inputs, num_outputs

# Define a simple structure to represent the neural network
mutable struct NNet
    weights::Array{Array{Float64, 2}}
    biases::Array{Array{Float64, 1}}
    numLayers::Int
    layerSizes::Array{Int}
    inputSize::Int
    outputSize::Int
    mins::Array{Float64}
    maxes::Array{Float64}
    means::Array{Float64}
    ranges::Array{Float64}

    # Constructor for loading the network from a file
    function NNet(file::String)
        # Open the file and skip comment lines
        f = open(file, "r")
        while startswith(peekline(f), "//")
            readline(f)
        end

        # Read basic network information
        metadata = split(readline(f), ",")
        numLayers = parse(Int, metadata[1])
        inputSize = parse(Int, metadata[2])
        outputSize = parse(Int, metadata[3])

        # Read layer sizes
        layerSizes = parse.(Int, split(readline(f), ","))

        # Skip a line
        readline(f)

        # Read input mins, maxes, means, and ranges
        mins = parse.(Float64, split(readline(f), ","))
        maxes = parse.(Float64, split(readline(f), ","))
        means = parse.(Float64, split(readline(f), ","))
        ranges = parse.(Float64, split(readline(f), ","))

        # Initialize weights and biases as empty arrays
        weights = []
        biases = []

        # Read weights and biases for each layer
        for layer in 1:numLayers
            weight_matrix = Array{Float64}(undef, layerSizes[layer+1], layerSizes[layer])
            for i in 1:layerSizes[layer+1]
                weight_matrix[i, :] = parse.(Float64, split(readline(f), ","))
            end
            push!(weights, weight_matrix)

            bias_vector = parse.(Float64, split(readline(f), ","))
            push!(biases, bias_vector)
        end

        close(f)

        return new(weights, biases, numLayers, layerSizes, inputSize, outputSize, mins, maxes, means, ranges)
    end
end

# Function to evaluate the network for one set of inputs
function evaluate_network(nnet::NNet, input::Array{Float64, 1})
    inputs = Array{Float64}(undef, nnet.inputSize)
    
    # Normalize inputs
    for i in 1:nnet.inputSize
        if input[i] < nnet.mins[i]
            inputs[i] = (nnet.mins[i] - nnet.means[i]) / nnet.ranges[i]
        elseif input[i] > nnet.maxes[i]
            inputs[i] = (nnet.maxes[i] - nnet.means[i]) / nnet.ranges[i]
        else
            inputs[i] = (input[i] - nnet.means[i]) / nnet.ranges[i]
        end
    end

    # Perform forward pass through the network
    for layer in 1:nnet.numLayers-1
        inputs = max.(nnet.weights[layer] * inputs .+ nnet.biases[layer], 0)
    end
    outputs = nnet.weights[end] * inputs .+ nnet.biases[end]

    # Undo output normalization
    for i in 1:nnet.outputSize
        outputs[i] = outputs[i] * nnet.ranges[end] + nnet.means[end]
    end

    return outputs
end

# Function to evaluate the network for multiple sets of inputs
function evaluate_network_multiple(nnet::NNet, input::Array{Float64, 2})
    numInputs = size(input, 2)
    inputs = Array{Float64}(undef, nnet.inputSize, numInputs)

    # Normalize inputs
    for i in 1:nnet.inputSize
        for j in 1:numInputs
            if input[i, j] < nnet.mins[i]
                inputs[i, j] = (nnet.mins[i] - nnet.means[i]) / nnet.ranges[i]
            elseif input[i, j] > nnet.maxes[i]
                inputs[i, j] = (nnet.maxes[i] - nnet.means[i]) / nnet.ranges[i]
            else
                inputs[i, j] = (input[i, j] - nnet.means[i]) / nnet.ranges[i]
            end
        end
    end

    # Perform forward pass through the network
    for layer in 1:nnet.numLayers-1
        inputs = max.(nnet.weights[layer] * inputs .+ nnet.biases[layer], 0)
    end
    outputs = nnet.weights[end] * inputs .+ nnet.biases[end]

    # Undo output normalization
    for i in 1:nnet.outputSize, j in 1:numInputs
        outputs[i, j] = outputs[i, j] * nnet.ranges[end] + nnet.means[end]
    end

    return outputs
end

# Get the number of inputs in the network
function num_inputs(nnet::NNet)
    return nnet.inputSize
end

# Get the number of outputs in the network
function num_outputs(nnet::NNet)
    return nnet.outputSize
end

end # module
