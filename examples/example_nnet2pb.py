from NNet.converters.nnet2pb import nnet2pb

# nnet file to convert to onnx
nnetFile = "../nnet/TestNetwork.nnet"

# Tensorflow pb file
pbFile = '../nnet/TestNetwork2.pb'

# Convert the file
nnet2pb(nnetFile, pbFile)