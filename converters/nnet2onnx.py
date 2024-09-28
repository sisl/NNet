import numpy as np
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar="y_out", inputVar="X", normalizeNetwork=False):
    """
    Convert a .nnet file to onnx format
    Args:
        nnetFile: (string) .nnet file to convert to onnx
        onnxFile: (string) Optional, name for the created .onnx file
        outputName: (string) Optional, name of the output variable in onnx
        normalizeNetwork: (bool) If true, adapt the network weights and biases so that 
                                 networks and inputs do not need to be normalized. Default is False.
    """
    # Normalize the network or read it
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    # Default onnx filename if none specified
    if not onnxFile:
        onnxFile = f"{nnetFile[:-4]}.onnx"

    # Initialize graph
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []

    # Loop through each layer of the network and add operations and initializers
    for i in range(numLayers):
        # Use outputVar for the last layer
        outputName = f"H{i}"
        if i == numLayers - 1:
            outputName = outputVar

        # Weight matrix multiplication
        operations.append(helper.make_node("MatMul", [f"W{i}", inputVar], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        # Bias addition
        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        # Use ReLU activation for all layers except the last layer
        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [f"H{i}"], [f"R{i}"]))
            inputVar = f"R{i}"

    # Create the graph and model in onnx
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    # Print statements
    print(f"Converted NNet model at {nnetFile}")
    print(f"    to an ONNX model at {onnxFile}")

    # Save the ONNX model
    onnx.save(model_def, onnxFile)

if __name__ == '__main__':
    # Read user inputs and run nnet2onnx function
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        onnxFile = sys.argv[2] if len(sys.argv) > 2 else ""
        outputVar = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2onnx(nnetFile, onnxFile, outputVar)
    else:
        print("Need to specify which .nnet file to convert to ONNX!")
