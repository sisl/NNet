import numpy as np
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar="y_out", inputVar="X", normalizeNetwork=False):
    """
    Convert a .nnet file to ONNX format.

    Args:
        nnetFile (str): .nnet file to convert to ONNX.
        onnxFile (str, optional): Name for the created .onnx file. Defaults to the name of the input file with an .onnx extension.
        outputVar (str, optional): Name of the output variable in ONNX. Default is "y_out".
        inputVar (str, optional): Name of the input variable in ONNX. Default is "X".
        normalizeNetwork (bool, optional): If True, adapt the network weights and biases so that networks and inputs do not need to be normalized. Default is False.
    """
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except Exception as e:
        print(f"Error reading NNet file: {e}")
        return

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    # Default ONNX filename if none specified
    if not onnxFile:
        onnxFile = f"{nnetFile[:-4]}.onnx"

    # Initialize graph inputs and outputs
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [None, inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [None, outputSize])]
    operations = []
    initializers = []

    # Loop through each layer of the network and add operations and initializers
    for i in range(numLayers):
        # Use outputVar for the last layer
        outputName = f"H{i}" if i < numLayers - 1 else outputVar

        # Weight matrix multiplication
        operations.append(helper.make_node("MatMul", [inputVar if i == 0 else f"R{i-1}", f"W{i}"], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        # Bias addition
        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        # Use ReLU activation for all layers except the last layer
        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [f"H{i}"], [f"R{i}"]))

    # Create the graph and model in ONNX
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    # Print statements
    print(f"Converted NNet model at {nnetFile} to an ONNX model at {onnxFile}")

    # Save the ONNX model
    try:
        onnx.save(model_def, onnxFile)
        print(f"ONNX model saved successfully at {onnxFile}")
    except Exception as e:
        print(f"Error saving the ONNX model: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        onnxFile = sys.argv[2] if len(sys.argv) > 2 else ""
        outputVar = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2onnx(nnetFile, onnxFile, outputVar)
    else:
        print("Error: Need to specify the .nnet file to convert to ONNX!")
