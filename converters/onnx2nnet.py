import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    """
    Write a .nnet file from an onnx file.
    
    Args:
        onnxFile (str): Path to onnx file.
        inputMins (list, optional): Minimum values for each neural network input.
        inputMaxes (list, optional): Maximum values for each neural network input.
        means (list, optional): Mean value for each input and value for the mean of all outputs (for normalization).
        ranges (list, optional): Range value for each input and value for the range of all outputs (for normalization).
        inputName (str, optional): Name of operation corresponding to input.
        outputName (str, optional): Name of operation corresponding to output.
    """

    if nnetFile == "":
        nnetFile = f"{onnxFile[:-4]}nnet"

    model = onnx.load(onnxFile)
    graph = model.graph

    if not inputName:
        assert len(graph.input) == 1, "Graph should have exactly one input!"
        inputName = graph.input[0].name
    if not outputName:
        assert len(graph.output) == 1, "Graph should have exactly one output!"
        outputName = graph.output[0].name

    weights = []
    biases = []

    # Loop through nodes in the graph
    for node in graph.node:
        if inputName in node.input:
            if node.op_type == "MatMul":
                assert len(node.input) == 2
                
                # Extract the weight matrix
                weightName = node.input[1] if node.input[0] == inputName else node.input[0]
                weights += [numpy_helper.to_array(init) for init in graph.initializer if init.name == weightName]

                # Update inputName
                inputName = node.output[0]

            elif node.op_type == "Add":
                assert len(node.input) == 2

                # Extract the bias vector
                biasName = node.input[1] if node.input[0] == inputName else node.input[0]
                biases += [numpy_helper.to_array(init) for init in graph.initializer if init.name == biasName]

                # Update inputName
                inputName = node.output[0]

            elif node.op_type == "Relu":
                inputName = node.output[0]

            else:
                print(f"Node operation type {node.op_type} not supported!")
                weights, biases = [], []
                break

            if outputName == inputName:
                break

    # Verify extraction and write the .nnet file
    if outputName == inputName and len(weights) > 0 and len(weights) == len(biases):
        inputSize = weights[0].shape[0]

        # Set defaults if not provided
        inputMins = inputMins if inputMins is not None else inputSize * [np.finfo(np.float32).min]
        inputMaxes = inputMaxes if inputMaxes is not None else inputSize * [np.finfo(np.float32).max]
        means = means if means is not None else (inputSize + 1) * [0.0]
        ranges = ranges if ranges is not None else (inputSize + 1) * [1.0]

        # Print success
        print(f"Converted ONNX model at {onnxFile} to an NNet model at {nnetFile}")

        # Write NNet file
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)

    else:
        print("Could not write NNet file due to an error in extraction!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("WARNING: Using default values for input bounds and normalization constants.")
        onnxFile = sys.argv[1]
        if len(sys.argv) > 2:
            nnetFile = sys.argv[2]
            onnx2nnet(onnxFile, nnetFile=nnetFile)
        else:
            onnx2nnet(onnxFile)
    else:
        print("Need to specify which ONNX file to convert to .nnet!")
