from NNet.converters.nnet2onnx import nnet2onnx

# nnet file to convert to onnx
nnetFile = "../nnet/TestNetwork.nnet"

# ONNX file
onnxFile = '../nnet/TestNetwork2.onnx'

# Convert the file
nnet2onnx(nnetFile, onnxFile)