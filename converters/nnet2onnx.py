import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar="y_out", inputVar="X", normalizeNetwork=False):
    """
    Convert a .nnet file to ONNX format.
    """
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except Exception as e:
        print(f"Error reading NNet file: {e}")
        return

    # Ensure weights and biases arrays are valid
    if len(weights) == 0 or len(biases) == 0:
        print(f"Error: Weights or biases are empty in {nnetFile}")
        return

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    print(f"Input size: {inputSize}, Output size: {outputSize}, Number of layers: {numLayers}")

    # Default ONNX filename if none specified
    if not onnxFile:
        onnxFile = f"{nnetFile[:-5]}_model.onnx"  # Changed to avoid the double dot issue

    # Initialize graph inputs and outputs
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [None, inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [None, outputSize])]
    operations = []
    initializers = []

    # Loop through each layer of the network and add operations and initializers
    for i in range(numLayers):
        # Print debug information about the layer's weight and bias shapes
        print(f"Layer {i}: Weight shape {weights[i].shape}, Bias shape {biases[i].shape}")

        # Ensure dimensions match for matrix multiplication
        if i == 0 and weights[i].shape[1] != inputSize:
            print(f"Error: Shape mismatch at input layer. Expected {inputSize}, got {weights[i].shape[1]}")
            return
        elif i > 0 and weights[i].shape[1] != weights[i - 1].shape[0]:
            print(f"Error: Shape mismatch between layers {i-1} and {i} in weights.")
            return

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
