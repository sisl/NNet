include("nnet.jl")

nnet = NNet("../../nnet/TestNetwork.nnet")
println(evaluate_network(nnet,[15299.0,-1.142,-1.142,600.0,500.0]))
