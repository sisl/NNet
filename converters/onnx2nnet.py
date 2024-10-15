import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    """
    Convert an ONNX model to .nnet format.

    Args:
        onnxFile (str): Path to the ONNX file.
        inputMins (list): Optional, minimum values for each input.
        inputMaxes (list): Optional, maximum values for each input.
        means (list): Optional, mean values for normalization.
        ranges (list): Optional, range values for normalization.
        nnetFile (str): Optional, name of the output .nnet file.
        inputName (str): Optional, name of the input node.
        outputName (str): Optional, name of the output node.
    """
    if not nnetFile:
        nnetFile = onnxFile.replace('.onnx', '.nnet')

    model = onnx.load(onnxFile)
    graph = model.graph

    if not inputName:
        assert len(graph.input) == 1, "ONNX graph must have exactly one input."
        inputName = graph.input[0].name

    if not outputName:
        assert len(graph.output) == 1, "ONNX graph must have exactly one output."
        outputName = graph.output[0].name

    weights, biases = extract_weights_and_biases(graph, inputName, outputName)

    if weights and biases and len(weights) == len(biases):
        inputSize = weights[0].shape[1]

        inputMins = inputMins or [float('-inf')] * inputSize
        inputMaxes = inputMaxes or [float('inf')] * inputSize
        means = means or [0.0] * inputSize
        ranges = ranges or [1.0] * inputSize

        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
        print(f"ONNX model converted successfully to {nnetFile}.")
    else:
        raise ValueError("Failed to extract valid weights and biases from the ONNX model.")

def extract_weights_and_biases(graph, inputName, outputName):
    weights = []
    biases = []
    for node in graph.node:
        if node.op_type == "MatMul":
            weights.append(_get_weights(graph, node))
        elif node.op_type == "Add":
            biases.append(_get_biases(graph, node))
    return weights, biases

def _get_weights(graph, node):
    for initializer in graph.initializer:
        if initializer.name == node.input[1]:
            return numpy_helper.to_array(initializer)
    raise ValueError(f"Could not find weights for node {node.name}")

def _get_biases(graph, node):
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
