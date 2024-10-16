import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar="y_out", inputVar="X", normalizeNetwork=False):
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    if onnxFile == "":
        onnxFile = nnetFile[:-4] + 'onnx'

    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []

    currentInputSize = inputSize  # Track input size for each layer

    for i in range(numLayers):
        outputName = f"H{i}" if i != numLayers - 1 else outputVar

        # Ensure the current layer's weights are compatible
        if weights[i].shape[1] != currentInputSize:
            raise ValueError(f"Weight matrix at layer {i} has incompatible shape: {weights[i].shape}. Expected input size {currentInputSize}.")

        operations.append(helper.make_node("MatMul", [f"W{i}", inputVar], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [outputName], [f"R{i}"]))
            inputVar = f"R{i}"
            currentInputSize = weights[i].shape[0]  # Update input size for the next layer

    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)
    onnx.save(model_def, onnxFile)
    print(f"Converted NNet model at {nnetFile} to ONNX model at {onnxFile}")
