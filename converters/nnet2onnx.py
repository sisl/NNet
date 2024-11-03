import numpy as np
import sys
import onnx
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
    normalizeNetwork: bool = False
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
    # Initialize weights and biases as empty lists to handle cases where they might not be assigned
    weights, biases = [], []
    
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except FileNotFoundError:
        print(f"Error: The file {nnetFile} was not found.")
        raise  # Re-raise the exception to allow the test to catch it

    if not weights or not biases:  # Check if weights and biases are populated
        print(f"Error: The file {nnetFile} could not be parsed correctly.")
        raise ValueError("Parsing error: weights and biases could not be determined.")

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
    for i in range(numLayers):
        outputName = f"H{i}"
        if i == numLayers - 1:
            outputName = outputVar

        operations.append(helper.make_node("MatMul", [inputVar, f"W{i}"], [f"M{i}"]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name=f"W{i}"))

        operations.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name=f"B{i}"))

        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", [outputName], [f"R{i}"]))
            inputVar = f"R{i}"

    # Create the graph and model in ONNX format
    graph_proto = helper.make_graph(operations, "nnet2onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)

    # Save the ONNX model to file
    onnx.save(model_def, onnxFile)
    print(f"Converted NNet model at {nnetFile} to an ONNX model at {onnxFile}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert a .nnet file to ONNX format.")
    parser.add_argument("nnetFile", type=str, help="The .nnet file to convert")
    parser.add_argument("--onnxFile", type=str, default="", help="Optional: Name of the output ONNX file")
    parser.add_argument("--outputVar", type=str, default="y_out", help="Optional: Name of the output variable")
    parser.add_argument("--inputVar", type=str, default="X", help="Optional: Name of the input variable")
    parser.add_argument("--normalize", action="store_true", help="Normalize network weights and biases")

    args = parser.parse_args()

    # Call the nnet2onnx function with parsed arguments
    nnet2onnx(
        nnetFile=args.nnetFile, 
        onnxFile=args.onnxFile, 
        outputVar=args.outputVar, 
        inputVar=args.inputVar, 
        normalizeNetwork=args.normalize
    )

if __name__ == "__main__":
    main()
