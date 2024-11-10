import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet


def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    """
    Convert an ONNX file to a .nnet file.
    
    Args:
        onnxFile (str): Path to the ONNX file.
        inputMins (list, optional): Minimum values for each input. Default: None.
        inputMaxes (list, optional): Maximum values for each input. Default: None.
        means (list, optional): Mean values for inputs and output normalization. Default: None.
        ranges (list, optional): Range values for inputs and output normalization. Default: None.
        nnetFile (str, optional): Output .nnet file path. Default: "".
        inputName (str, optional): Name of the input tensor in the ONNX model. Default: "".
        outputName (str, optional): Name of the output tensor in the ONNX model. Default: "".
    """
    if nnetFile == "":
        nnetFile = onnxFile.replace(".onnx", ".nnet")

    # Load the ONNX model and extract the graph
    try:
        model = onnx.load(onnxFile)
        graph = model.graph
    except Exception as e:
        print(f"Error loading ONNX file '{onnxFile}': {e}")
        return

    # Automatically detect input and output names if not provided
    if not inputName:
        if len(graph.input) != 1:
            raise ValueError("Multiple inputs found in the ONNX model. Please specify the input name explicitly.")
        inputName = graph.input[0].name
    if not outputName:
        if len(graph.output) != 1:
            raise ValueError("Multiple outputs found in the ONNX model. Please specify the output name explicitly.")
        outputName = graph.output[0].name

    weights = []
    biases = []

    # Debugging information
    print(f"Input Name: {inputName}")
    print(f"Output Name: {outputName}")
    print("Processing ONNX graph nodes...")

    # Loop through the nodes in the ONNX graph to extract weights and biases
    for node in graph.node:
        print(f"Node: {node.name}, Operation: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")
        if inputName in node.input:
            if node.op_type == "MatMul":
                weightName = next((inp for inp in node.input if inp != inputName), None)
                if not weightName:
                    print(f"Error: Weight tensor not found for node {node.name}.")
                    return
                try:
                    weight = next(numpy_helper.to_array(init) for init in graph.initializer if init.name == weightName)
                    weights.append(weight)
                except StopIteration:
                    print(f"Error: Initializer for weight '{weightName}' not found in ONNX graph.")
                    return

                # Update the inputName to the output of the current node
                inputName = node.output[0]

            elif node.op_type == "Add":
                biasName = next((inp for inp in node.input if inp != inputName), None)
                if not biasName:
                    print(f"Error: Bias tensor not found for node {node.name}.")
                    return
                try:
                    bias = next(numpy_helper.to_array(init) for init in graph.initializer if init.name == biasName)
                    biases.append(bias)
                except StopIteration:
                    print(f"Error: Initializer for bias '{biasName}' not found in ONNX graph.")
                    return

                # Update the inputName to the output of the current node
                inputName = node.output[0]

            elif node.op_type == "Relu":
                # ReLU activations are implicit in .nnet format
                inputName = node.output[0]

            else:
                print(f"Unsupported node operation: {node.op_type}. Skipping node.")
                return  # Exit if an unsupported operation is encountered

            if inputName == outputName:
                break  # Stop once the output node is reached

    # Validation and writing the .nnet file
    if len(weights) == len(biases) > 0 and inputName == outputName:
        inputSize = weights[0].shape[1]

        # Set default normalization values if not provided
        inputMins = inputMins or [np.finfo(np.float32).min] * inputSize
        inputMaxes = inputMaxes or [np.finfo(np.float32).max] * inputSize
        means = means or [0.0] * (inputSize + 1)
        ranges = ranges or [1.0] * (inputSize + 1)

        print(f"Converted ONNX model '{onnxFile}' to NNet model '{nnetFile}'.")
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
    else:
        print("Error: Unable to extract weights and biases or invalid network structure.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("WARNING: Using default input bounds and normalization constants.")
        onnxFile = sys.argv[1]
        nnetFile = sys.argv[2] if len(sys.argv) > 2 else ""
        onnx2nnet(onnxFile, nnetFile=nnetFile)
    else:
        print("Usage: python onnx2nnet.py <onnxFile> [<nnetFile>]")
