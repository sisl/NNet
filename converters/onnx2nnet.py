import numpy as np
import sys
import onnx
from onnx import numpy_helper
from NNet.utils.writeNNet import writeNNet

def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    '''
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
    '''

    if nnetFile == "":
        nnetFile = onnxFile.replace(".onnx", ".nnet")

    # Load the ONNX model and extract the graph
    model = onnx.load(onnxFile)
    graph = model.graph

    # Automatically detect input and output names if not provided
    if not inputName:
        assert len(graph.input) == 1, "Multiple inputs found. Specify the input name."
        inputName = graph.input[0].name
    if not outputName:
        assert len(graph.output) == 1, "Multiple outputs found. Specify the output name."
        outputName = graph.output[0].name

    weights = []
    biases = []

    # Loop through the nodes in the ONNX graph to extract weights and biases
    for node in graph.node:
        if inputName in node.input:
            if node.op_type == "MatMul":
                weightName = next(inp for inp in node.input if inp != inputName)
                weight = next(numpy_helper.to_array(init) for init in graph.initializer if init.name == weightName)
                weights.append(weight)

                # Update the inputName to the output of the current node
                inputName = node.output[0]

            elif node.op_type == "Add":
                biasName = next(inp for inp in node.input if inp != inputName)
                bias = next(numpy_helper.to_array(init) for init in graph.initializer if init.name == biasName)
                biases.append(bias)

                # Update the inputName to the output of the current node
                inputName = node.output[0]

            elif node.op_type == "Relu":
                # For .nnet format, ReLU activations are implicit
                inputName = node.output[0]

            else:
                print(f"Unsupported node operation: {node.op_type}")
                return  # Exit if an unsupported operation is encountered

            if inputName == outputName:
                break  # Stop once the output node is reached

    if len(weights) == len(biases) > 0 and inputName == outputName:
        inputSize = weights[0].shape[0]

        # Set default normalization values if not provided
        inputMins = inputMins or [np.finfo(np.float32).min] * inputSize
        inputMaxes = inputMaxes or [np.finfo(np.float32).max] * inputSize
        means = means or [0.0] * (inputSize + 1)
        ranges = ranges or [1.0] * (inputSize + 1)

        print(f"Converted ONNX model '{onnxFile}' to NNet model '{nnetFile}'.")

        # Write the .nnet file
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
