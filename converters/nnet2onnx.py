import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet
import argparse
from typing import Optional


def nnet2onnx(
    nnetFile: str,
    onnxFile: Optional[str] = "",
    outputVar: str = "y_out",
    inputVar: str = "X",
    normalizeNetwork: bool = False,
) -> None:
    """
    Convert a .nnet file to ONNX format.

    Args:
        nnetFile (str): Path to the .nnet file to convert.
        onnxFile (Optional[str]): Optional, name for the created .onnx file. Defaults to the same name as the .nnet file.
        outputVar (str): Optional, name of the output variable in ONNX. Defaults to 'y_out'.
        inputVar (str): Name of the input variable in ONNX. Defaults to 'X'.
        normalizeNetwork (bool): If True, adapt the network weights and biases so that networks and inputs
                                 do not need normalization. Defaults to False.
    """
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file {nnetFile} was not found.") from e

    # Validate input shapes across layers
    for i, (w, b) in enumerate(zip(weights, biases)):
        if w.shape[0] != len(b):
            raise ValueError(f"Mismatch at Layer {i}: Weight rows {w.shape[0]} != Bias length {len(b)}")

    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)

    # Default ONNX filename if none specified
    if not onnxFile:
        onnxFile = f"{nnetFile[:-5]}.onnx"

    # Initialize the graph
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []

    # Build the ONNX model layer by layer
    currentInput = inputVar  # Track the current input variable
    for i, (w, b) in enumerate(zip(weights, biases)):
        print(f"Layer {i}: Weights shape: {w.shape}, Biases shape: {b.shape}")

        # Define layer names
        weightName = f"W{i}"
        biasName = f"B{i}"
        matmulOutput = f"M{i}"
        outputName = outputVar if i == numLayers - 1 else f"H{i}"

        # Add MatMul and Add operations
        operations.append(helper.make_node("MatMul", [currentInput, weightName], [matmulOutput]))
        operations.append(helper.make_node("Add", [matmulOutput, biasName], [outputName]))
        
        # Update current input for next layer
        currentInput = outputName

        # Save weights and biases as initializers
        initializers.append(numpy_helper.from_array(w.astype(np.float32), weightName))
        initializers.append(numpy_helper.from_array(b.astype(np.float32), biasName))

        # Apply ReLU activation except for the final layer
        if i < numLayers - 1:
            reluOutput = f"R{i}"
            operations.append(helper.make_node("Relu", [outputName], [reluOutput]))
            currentInput = reluOutput

    # Create the graph and model
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    # Save the ONNX model to file
    onnx.save(model_def, onnxFile)
    print(f"Converted NNet model at {nnetFile} to an ONNX model at {onnxFile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .nnet file to ONNX format.")
    parser.add_argument("nnetFile", type=str, help="The .nnet file to convert")
    parser.add_argument("--onnxFile", type=str, default="", help="Optional: Name of the output ONNX file")
    parser.add_argument("--outputVar", type=str, default="y_out", help="Optional: Name of the output variable")
    parser.add_argument("--inputVar", type=str, default="X", help="Optional: Name of the input variable")
    parser.add_argument("--normalize", action="store_true", help="Normalize network weights and biases")
    args = parser.parse_args()

    try:
        nnet2onnx(
            nnetFile=args.nnetFile,
            onnxFile=args.onnxFile,
            outputVar=args.outputVar,
            inputVar=args.inputVar,
            normalizeNetwork=args.normalize,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
