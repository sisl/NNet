include("nnet.jl")

nnet = NNet("../nnet/TestNetwork.nnet")
println("Num Inputs: "*string(num_inputs(nnet)))
println("Num Outputs: "*string(num_outputs(nnet)))
println("One evaluation:")
println(evaluate_network(nnet,[15299.0,-1.142,-1.142,600.0,500.0]))
println("\nMultiple evaluations at once:")
println(evaluate_network_multiple(nnet,[[15299.0, -1.142, -1.142, 600.0, 500.0] [15299.0,-1.142,-1.142,600.0,500.0]]))
