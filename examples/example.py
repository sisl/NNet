import sys
from NNet.python.nnet import NNet  
sys.path.append('../..')
nnet = NNet('../nnet/TestNetwork.nnet')
print(f"Num Inputs: {nnet.num_inputs()}")
print(f"Num Outputs: {nnet.num_outputs()}")
print("One evaluation:")
single_evaluation = nnet.evaluate_network([15299.0, 0.0, -3.1, 600.0, 500.0])
print(single_evaluation)
print("\nMultiple evaluations at once:")
multiple_evaluations = nnet.evaluate_network_multiple([
    [15299.0, 0.0, -3.1, 600.0, 500.0],
    [15299.0, 0.0, -3.1, 600.0, 500.0]
])
print(multiple_evaluations)
