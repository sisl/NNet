import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnx_file, input_mins=None, input_maxes=None, means=None, ranges=None, nnet_file="", input_name="", output_name=""):
    '''
    Convert an ONNX model to .nnet format
    
    Args:
        onnx_file (str): Path to the ONNX file.
        input_mins (list): Optional, minimum values for each neural network input.
        input_maxes (list): Optional, maximum values for each neural network input.
        means (list): Optional, mean values for normalization of inputs/outputs.
        ranges (list): Optional, range values for normalization of inputs/outputs.
        nnet_file (str): Optional, name for the output .nnet file.
        input_name (str): Optional, name of the input operation.
        output_name (str): Optional, name of the output operation.
    '''
    if not nnet_file:
        nnet_file = onnx_file.replace('.onnx', '.nnet')

    # Load ONNX model
    model = onnx.load(onnx_file)
    graph = model.graph

    # Set input and output names if not provided
    if not input_name:
        assert len(graph.input) == 1, "Graph should have one input."
        input_name = graph.input[0].name

    if not output_name:
        assert len(graph.output) == 1, "Graph should have one output."
        output_name = graph.output[0].name

    weights = []
    biases = []

    # Process nodes in the graph
    for node in graph.node:
        if input_name in node.input:
            if node.op_type == 'MatMul':
                weights.append(_get_weights(graph, node))
            elif node.op_type == 'Add':
                biases.append(_get_biases(graph, node))
            elif node.op_type == 'Relu':
                continue
            else:
                print(f"Node operation type {node.op_type} not supported!")
                return False

            # Check if we have reached the output node
            if output_name == node.output[0]:
                break

    # Ensure extracted weights and biases match
    if len(weights) > 0 and len(weights) == len(biases):
        input_size = weights[0].shape[0]

        # Set default values for input bounds and normalization
        if input_mins is None:
            input_mins = [np.finfo(np.float32).min] * input_size
        if input_maxes is None:
            input_maxes = [np.finfo(np.float32).max] * input_size
        if means is None:
            means = [0.0] * (input_size + 1)
        if ranges is None:
            ranges = [1.0] * (input_size + 1)

        print(f"Successfully converted ONNX model '{onnx_file}' to NNet format '{nnet_file}'")
        writeNNet(weights, biases, input_mins, input_maxes, means, ranges, nnet_file)
    else:
        print("Error: Could not extract weights and biases properly.")
        return False

    return True

def _get_weights(graph, node):
    ''' Helper function to extract weights from a MatMul node. '''
    for initializer in graph.initializer:
        if initializer.name == node.input[1]:
            return numpy_helper.to_array(initializer)
    raise ValueError(f"Could not find weights for node {node.name}")

def _get_biases(graph, node):
    ''' Helper function to extract biases from an Add node. '''
    for initializer in graph.initializer:
        if initializer.name == node.input[1]:
            return numpy_helper.to_array(initializer)
    raise ValueError(f"Could not find biases for node {node.name}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        onnx_file = sys.argv[1]
        nnet_file = sys.argv[2] if len(sys.argv) > 2 else ""
        print("WARNING: Using default values for input bounds and normalization.")
        onnx2nnet(onnx_file, nnet_file=nnet_file)
    else:
        print("Usage: python onnx2nnet.py <onnx_file> [<nnet_file>]")
