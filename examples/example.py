import sys
sys.path.append('../..')
from NNet.python.nnet import *
nnet = NNet('../nnet/TestNetwork.nnet')
print("Num Inputs: %d"%nnet.num_inputs())
print("Num Outputs: %d"%nnet.num_outputs())
print("One evaluation:")
print(nnet.evaluate_network([15299.0,0.0,-3.1,600.0,500.0]))
print("\nMultiple evaluations at once:")
print(nnet.evaluate_network_multiple([[15299.0,0.0,-3.1,600.0,500.0],[15299.0,0.0,-3.1,600.0,500.0]]))
