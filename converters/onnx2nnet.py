import numpy as np
import os
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    """
    Convert an ONNX model to .nnet format.

    Args:
        onnxFile (str): Path to the ONNX file.
        inputMins (list, optional): Minimum values for each neural network input.
        inputMaxes (list, optional): Maximum values for each neural network input.
        means (list, optional): Mean values for normalization.
        ranges (list, optional): Range values for normalization.
        nnetFile (str, optional): Output filename for the .nnet file.
        inputName (str, optional): Name of the input node.
        outputName (str, optional): Name of the output node.
    """

    # Default the nnetFile name if not provided
    if not nnetFile:
        nnetFile = onnxFile.replace('.onnx', '.nnet')

    # Load the ONNX model
    model = onnx.load(onnxFile)
    graph = model.graph

    # Set input and output names if not explicitly provided
    if not inputName:
        assert len(graph.input) == 1, "Graph must have exactly one input."
        inputName = graph.input[0].name

    if not outputName:
        assert len(graph.output) == 1, "Graph must have exactly one output."
        outputName = graph.output[0].name

    weights, biases = extract_weights_and_biases(graph, inputName, outputName)

    if weights and biases and len(weights) == len(biases):
        input_size = weights[0].shape[0]

        # Default input bounds and normalization values
        inputMins = inputMins or [np.finfo(np.float32).min] * input_size
        inputMaxes = inputMaxes or [np.finfo(np.float32).max] * input_size
        means = means or [0.0] * (input_size + 1)
        ranges = ranges or [1.0] * (input_size + 1)

        print(f"Successfully converted {onnxFile} to {nnetFile}.")
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
    else:
        raise ValueError("Error: Failed to extract valid weights and biases from the ONNX model.")

def extract_weights_and_biases(graph, input_name, output_name):
    """
    Extract weights and biases from the ONNX model graph.

    Args:
        graph (onnx.GraphProto): The ONNX model graph.
        input_name (str): Name of the input node.
        output_name (str): Name of the output node.

    Returns:
        tuple: (weights, biases), lists of numpy arrays.
    """
    weights, biases = [], []

    for node in graph.node:
        if input_name in node.input:
            if node.op_type == 'MatMul':
                weights.append(get_initializer(graph, node.input[1]))
            elif node.op_type == 'Add':
                biases.append(get_initializer(graph, node.input[1]))
            elif node.op_type == 'Relu':
                continue
            else:
                raise ValueError(f"Unsupported node operation: {node.op_type}")

            if output_name in node.output:
                break

    return weights, biases

def get_initializer(graph, name):
    """
    Retrieve an initializer from the ONNX graph by name.

    Args:
        graph (onnx.GraphProto): The ONNX model graph.
        name (str): Name of the initializer to retrieve.

    Returns:
        numpy.ndarray: The initializer as a NumPy array.
    """
    for initializer in graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer)
    raise ValueError(f"Initializer '{name}' not found in the graph.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        onnxFile = sys.argv[1]
        nnetFile = sys.argv[2] if len(sys.argv) > 2 else ""
        print("Converting ONNX to NNet...")
        onnx2nnet(onnxFile, nnetFile=nnetFile)
    else:
        print("Usage: python onnx2nnet.py <onnx_file> [<nnet_file>]")
