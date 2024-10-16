import numpy as np
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar="y_out", inputVar="X", normalizeNetwork=False):
    '''
    Convert a .nnet file to ONNX format.

    Args:
        nnetFile: (string) .nnet file to convert to ONNX.
        onnxFile: (string) Optional, name for the output .onnx file.
        outputVar: (string) Optional, name of the output variable in ONNX.
        inputVar: (string) Optional, name of the input variable in ONNX.
        normalizeNetwork: (bool) Normalize network weights and biases if True. Default is False.
    '''

    # Read or normalize the network data
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    # Default ONNX filename if not specified
    if not onnxFile:
        onnxFile = nnetFile.replace('.nnet', '.onnx')

    # Initialize graph components
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []

    # Build the ONNX graph layer by layer
    for i in range(numLayers):
        layerOutput = f"H{i}"
        if i == numLayers - 1:
            layerOutput = outputVar  # Use outputVar for the final layer

        # Add MatMul operation
        matmul_output = f"M{i}"
        operations.append(helper.make_node("MatMul", [inputVar, f"W{i}"], [matmul_output]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        # Add Bias (Add operation)
        operations.append(helper.make_node("Add", [matmul_output, f"B{i}"], [layerOutput]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        # Add Relu activation (except for the last layer)
        if i < numLayers - 1:
            relu_output = f"R{i}"
            operations.append(helper.make_node("Relu", [layerOutput], [relu_output]))
            inputVar = relu_output  # Update inputVar for the next layer

    # Create the ONNX graph and model
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    # Print success messages
    print(f"Converted NNet model at {nnetFile} to an ONNX model at {onnxFile}")

    # Save the ONNX model to a file
    onnx.save(model_def, onnxFile)

if __name__ == '__main__':
    # Read user inputs from the command line
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        onnxFile = sys.argv[2] if len(sys.argv) > 2 else ""
        outputVar = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2onnx(nnetFile, onnxFile, outputVar)
    else:
        print("Usage: python nnet2onnx.py <nnetFile> [<onnxFile>] [<outputVar>]")
