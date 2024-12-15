from NNet.converters.onnx2nnet import onnx2nnet

# Min and max values used to bound the inputs
input_mins  = [0.0, -3.141593, -3.141593, 100.0, 0.0]
input_maxes = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]

# Mean and range values for normalizing the inputs and outputs
means  = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
ranges = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0, 373.94992]

# ONNX file to convert to .nnet file
onnx_file = '../nnet/TestNetwork2.onnx'

# Try to convert the ONNX file to .nnet format, with error handling
try:
    print(f"Converting {onnx_file} to .nnet format...")
    onnx2nnet(onnx_file, input_mins, input_maxes, means, ranges)
    print(f"Successfully converted {onnx_file} to .nnet format")
except Exception as e:
    print(f"Failed to convert {onnx_file}. Error: {e}")
