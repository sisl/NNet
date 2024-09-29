import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    """
    Write a .nnet file from an ONNX file.

    Args:
        onnxFile (str): Path to ONNX file.
        inputMins (list, optional): Minimum values for each neural network input.
        inputMaxes (list, optional): Maximum values for each neural network output.
        means (list, optional): Mean value for each input and value for mean of all outputs, used for normalization.
        ranges (list, optional): Range value for each input and value for range of all outputs, used for normalization.
        nnetFile (str, optional): Name of the output .nnet file. Defaults to input ONNX filename.
        inputName (str, optional): Name of the input operation. Defaults to the first input in ONNX graph.
        outputName (str, optional): Name of the output operation. Defaults to the first output in ONNX graph.
    """
    # Set default nnetFile name if none provided
    if not nnetFile:
        nnetFile = f"{onnxFile[:-4]}.nnet"

    try:
        model = onnx.load(onnxFile)
    except Exception as e:
        print(f"Error loading ONNX file: {e}")
        return

    graph = model.graph

    # Use default input/output names if not provided
    if not inputName:
        assert len(graph.input) == 1, "Graph should have exactly one input!"
        inputName = graph.input[0].name
    if not outputName:
        assert len(graph.output) == 1, "Graph should have exactly one output!"
        outputName = graph.output[0].name

    # Initialize lists for weights and biases
    weights = []
    biases = []

    # Loop through nodes in graph
    for node in graph.node:
        if inputName in node.input:
            if node.op_type == "MatMul":
                assert len(node.input) == 2, "MatMul node must have exactly 2 inputs."
                
                # Extract weight matrix
                weightName = node.input[1] if node.input[0] == inputName else node.input[0]
                weights += [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name == weightName]

                # Update input name to be the output of this node
                inputName = node.output[0]

            elif node.op_type == "Add":
                assert len(node.input) == 2, "Add node must have exactly 2 inputs."
                
                # Extract bias vector
                biasName = node.input[1] if node.input[0] == inputName else node.input[0]
                biases += [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name == biasName]

                # Update input name to be the output of this node
                inputName = node.output[0]

            elif node.op_type == "Relu":
                # Relu does not affect weights/biases, just update the input
                inputName = node.output[0]

            else:
                print(f"Node operation type {node.op_type} not supported!")
                return

            # Terminate once we find the output node
            if outputName == inputName:
                break

    # Check if weights and biases were successfully extracted
    if outputName == inputName and len(weights) == len(biases) > 0:
        inputSize = weights[0].shape[0]

        # Set default values for input bounds and normalization constants if not provided
        inputMins = inputMins if inputMins is not None else [np.finfo(np.float32).min] * inputSize
        inputMaxes = inputMaxes if inputMaxes is not None else [np.finfo(np.float32).max] * inputSize
        means = means if means is not None else [0.0] * (inputSize + 1)
        ranges = ranges if ranges is not None else [1.0] * (inputSize + 1)

        # Write the NNet file
        try:
            writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
            print(f"Converted ONNX model at {onnxFile} to NNet model at {nnetFile}")
        except Exception as e:
            print(f"Error writing NNet file: {e}")
    else:
        print("Could not convert ONNX file to NNet: mismatch in weights and biases.")

if __name__ == '__main__':
    # Read user inputs and run onnx2nnet function
    if len(sys.argv) > 1:
        print("WARNING: Using the default values of input bounds and normalization constants")
        onnxFile = sys.argv[1]
        nnetFile = sys.argv[2] if len(sys.argv) > 2 else ""
        onnx2nnet(onnxFile, nnetFile=nnetFile)
    else:
        print("Need to specify which ONNX file to convert to .nnet!")
