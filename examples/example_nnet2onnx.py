from NNet.converters.nnet2onnx import nnet2onnx

# Define the paths for the NNet file and the ONNX file
nnet_file = "../nnet/TestNetwork.nnet"
onnx_file = "../nnet/TestNetwork2.onnx"

# Try to convert the NNet file to ONNX format, with error handling
try:
    print(f"Converting {nnet_file} to {onnx_file}...")
    nnet2onnx(nnet_file, onnx_file)
    print(f"Successfully converted {nnet_file} to {onnx_file}")
except Exception as e:
    print(f"Failed to convert {nnet_file} to {onnx_file}. Error: {e}")
