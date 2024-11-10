import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet
from typing import Optional


def nnet2onnx(
    nnetFile: str,
    onnxFile: Optional[str] = None,
    outputVar: str = "y_out",
    inputVar: str = "X",
    normalizeNetwork: bool = False,
) -> None:
    """
    Convert a .nnet file to ONNX format.

    Args:
        nnetFile (str): Path to the .nnet file to convert.
        onnxFile (Optional[str]): Name for the created .onnx file. Defaults to the same name as the .nnet file.
        outputVar (str): Name of the output variable in ONNX. Defaults to 'y_out'.
        inputVar (str): Name of the input variable in ONNX. Defaults to 'X'.
        normalizeNetwork (bool): If True, normalize network weights and biases. Defaults to False.
    """
    # Validate input file extension
    if not nnetFile.endswith(".nnet"):
        raise ValueError(f"Input file must have a .nnet extension. Got: {nnetFile}")

    # Default ONNX file name
    if not onnxFile:
        onnxFile = nnetFile.replace(".nnet", ".onnx")

    try:
        # Read weights and biases from the .nnet file
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except FileNotFoundError:
        print(f"Error: The file {nnetFile} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading .nnet file: {e}")
        sys.exit(1)

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    # Validate dimensions and log layer info
    for i, (w, b) in enumerate(zip(weights, biases)):
        if w.shape[1] != (weights[i - 1].shape[0] if i > 0 else inputSize):
            raise ValueError(
                f"Dimension mismatch at layer {i}: Weights {w.shape}, expected input size {w.shape[1]}"
            )
        if b.shape[0] != w.shape[0]:
            raise ValueError(
                f"Bias dimension mismatch at layer {i}: Biases {b.shape}, expected {w.shape[0]}"
            )
        print(f"Layer {i}: Weights {w.shape}, Biases {b.shape}")

    # Create ONNX graph components
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [None, inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [None, outputSize])]
    nodes = []
    initializers = []

    currentInput = inputVar

    # Construct ONNX graph layer by layer
    for i in range(numLayers):
        weightName = f"W{i}"
        biasName = f"B{i}"
        matmulName = f"M{i}"
        addName = f"H{i}" if i < numLayers - 1 else outputVar

        # Add weight and bias initializers
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=weightName))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=biasName))

        # Add MatMul and Add nodes
        nodes.append(helper.make_node("MatMul", [currentInput, weightName], [matmulName]))
        nodes.append(helper.make_node("Add", [matmulName, biasName], [addName]))

        # Apply ReLU activation for all layers except the last
        if i < numLayers - 1:
            reluName = f"R{i}"
            nodes.append(helper.make_node("Relu", [addName], [reluName]))
            currentInput = reluName
        else:
            currentInput = addName

    # Create the ONNX graph and model
    graph = helper.make_graph(
        nodes=nodes,
        name="NNet_to_ONNX",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="nnet2onnx")

    # Validate and save the ONNX model
    try:
        onnx.checker.check_model(model)
        onnx.save_model(model, onnxFile)
        print(f"Successfully converted {nnetFile} to {onnxFile}")
    except Exception as e:
        print(f"Failed to save ONNX model: {e}")
        sys.exit(1)
