include("nnet.jl")

# Load the neural network
nnet = NNet("../nnet/TestNetwork.nnet")

# Print the number of inputs and outputs
println("Num Inputs: ", num_inputs(nnet))
println("Num Outputs: ", num_outputs(nnet))

# Perform one evaluation
println("One evaluation:")
result = evaluate_network(nnet, [15299.0, -1.142, -1.142, 600.0, 500.0])
println(result)

# Perform multiple evaluations at once
println("\nMultiple evaluations at once:")
multiple_results = evaluate_network_multiple(nnet, [
    [15299.0, -1.142, -1.142, 600.0, 500.0],
    [15299.0, -1.142, -1.142, 600.0, 500.0]
])
println(multiple_results)
