from NNet.converters.onnx2nnet import onnx2nnet

## Script showing how to run onnx2nnet
# Min and max values used to bound the inputs
inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# ONNX file to convert to .nnet file
onnxFile = '../nnet/TestNetwork2.onnx'

# Convert the file
onnx2nnet(onnxFile, inputMins, inputMaxes, means, ranges)