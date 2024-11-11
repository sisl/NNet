import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet


def nnet2onnx(
    nnetFile, onnxFile=None, outputVar="y_out", inputVar="X", normalizeNetwork=False
):
    if not nnetFile.endswith(".nnet"):
        raise ValueError(f"Input file must have a .nnet extension. Got: {nnetFile}")

    if not onnxFile:
        onnxFile = nnetFile.replace(".nnet", ".onnx")

    try:
        weights, biases = (
            normalizeNNet(nnetFile) if normalizeNetwork else readNNet(nnetFile)
        )
    except Exception as e:
        raise ValueError(f"Error reading NNet file: {e}")

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]

    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [None, inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [None, outputSize])]

    nodes = []
    initializers = []
    currentInput = inputVar

    for i, (w, b) in enumerate(zip(weights, biases)):
        weightName = f"W{i}"
        biasName = f"B{i}"
        matmulName = f"M{i}"
        addName = f"H{i}" if i < len(weights) - 1 else outputVar

        initializers.append(numpy_helper.from_array(w.astype(np.float32), name=weightName))
        initializers.append(numpy_helper.from_array(b.astype(np.float32), name=biasName))

        nodes.append(helper.make_node("MatMul", [currentInput, weightName], [matmulName]))
        nodes.append(helper.make_node("Add", [matmulName, biasName], [addName]))

        if i < len(weights) - 1:
            reluName = f"R{i}"
            nodes.append(helper.make_node("Relu", [addName], [reluName]))
            currentInput = reluName
        else:
            currentInput = addName

    graph = helper.make_graph(
        nodes=nodes,
        name="NNet_to_ONNX",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="nnet2onnx")
    onnx.save_model(model, onnxFile)
