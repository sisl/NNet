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
    """
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except FileNotFoundError:
        print(f"Error: The file {nnetFile} was not found.")
        sys.exit(1)

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
    for i, (w, b) in enumerate(zip(weights, biases)):
        print(f"Layer {i}: Weights shape: {w.shape}, Biases shape: {b.shape}")
        assert w.shape[1] == inputSize, f"Mismatch at Layer {i}: Expected {inputSize} inputs, got {w.shape[1]}"
        assert b.shape[0] == w.shape[0], f"Mismatch at Layer {i}: Expected {w.shape[0]} biases, got {b.shape[0]}"

        outputName = outputVar if i == numLayers - 1 else f"H{i}"

        operations.append(helper.make_node("MatMul", [inputVar, f"W{i}"], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(w.astype(np.float32), name=f"W{i}"))

        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(b.astype(np.float32), name=f"B{i}"))

        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [outputName], [f"R{i}"]))
            inputVar = f"R{i}"

    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)
    print(f"Converted NNet model at {nnetFile} to an ONNX model at {onnxFile}")
    onnx.save(model_def, onnxFile)


def main():
    parser = argparse.ArgumentParser(description="Convert a .nnet file to ONNX format.")
    parser.add_argument("nnetFile", type=str, help="The .nnet file to convert")
    parser.add_argument("--onnxFile", type=str, default="", help="Optional: Name of the output ONNX file")
    parser.add_argument("--outputVar", type=str, default="y_out", help="Optional: Name of the output variable")
    parser.add_argument("--inputVar", type=str, default="X", help="Optional: Name of the input variable")
    parser.add_argument("--normalize", action="store_true", help="Normalize network weights and biases")

    args = parser.parse_args()
    nnet2onnx(
        nnetFile=args.nnetFile,
        onnxFile=args.onnxFile,
        outputVar=args.outputVar,
        inputVar=args.inputVar,
        normalizeNetwork=args.normalize,
    )


if __name__ == "__main__":
    main()
