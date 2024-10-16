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

    # Validate the weights and biases to avoid errors in conversion
    if not weights or not biases:
        print(f"Error: Empty weights or biases found in {nnetFile}")
        return

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    print(f"Input size: {inputSize}, Output size: {outputSize}, Number of layers: {numLayers}")

    # Ensure valid ONNX filename
    if not onnxFile:
        onnxFile = f"{nnetFile[:-5]}.onnx"  # Avoids any '..' in filenames

    # Define input and output tensors for ONNX
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [None, inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [None, outputSize])]

    operations = []
    initializers = []

    for i in range(numLayers):
        print(f"Layer {i}: Weight shape {weights[i].shape}, Bias shape {biases[i].shape}")

        # Validate matrix multiplication dimensions
        if i > 0 and weights[i].shape[1] != weights[i - 1].shape[0]:
            print(f"Error: Shape mismatch between layer {i - 1} and {i}")
            return

        outputName = f"H{i}" if i < numLayers - 1 else outputVar

        # Add MatMul and Add operations for each layer
        operations.append(helper.make_node("MatMul", [inputVar if i == 0 else f"R{i-1}", f"W{i}"], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [f"H{i}"], [f"R{i}"]))

    # Create ONNX graph
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    try:
        onnx.save(model_def, onnxFile)
        print(f"ONNX model saved successfully at {onnxFile}")
    except Exception as e:
        print(f"Error saving the ONNX model: {e}")
