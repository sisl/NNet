import numpy as np
import onnx
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
import onnxruntime
from NNet.python.nnet import *

### Options###
nnetFile = "../nnet/TestNetwork.nnet"
testInput = np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32)
##############

# Convert NNET to ONNX and save ONNX network to given file
# Adapt network weights and biases so that no input or output normalization is required to evaluate network
onnxFile = nnetFile[:-4]+"onnx"
nnet2onnx(nnetFile,onnxFile=onnxFile,normalizeNetwork=True)

# Convert ONNX back to NNET and save NNET network
# Note that unless input mins and maxes are specified, the minimum and maximum floating point values will be written
nnetFile2 = nnetFile[:-4]+"v2.nnet"
onnx2nnet(onnxFile,nnetFile=nnetFile2)

## Test that the networks are equivalent
# Load models
nnet = NNet(nnetFile)
sess = onnxruntime.InferenceSession(onnxFile)
nnet2 = NNet(nnetFile2)

# Evaluate ONNX
onnxInputName = sess.get_inputs()[0].name
onnxOutputName = sess.get_outputs()[0].name
onnxEval = sess.run([onnxOutputName],{onnxInputName: testInput})[0]

# Evaluate Original NNET
inBounds = np.all(testInput>=nnet.mins) and np.all(testInput<=nnet.maxes)
if not inBounds: 
    print("WARNING: Test input is outside input bounds defined in NNet header!")
    print("Inputs are clipped before evaluation. so evaluations may differ")
    print("Test Input:  "+str(testInput))
    print("Input Mins:  "+str(nnet.mins))
    print("Input Maxes: "+str(nnet.maxes))
    print("")
nnetEval = nnet.evaluate_network(testInput)

# Evaluate New NNET
inBounds = np.all(testInput>=nnet2.mins) and np.all(testInput<=nnet2.maxes)
if not inBounds: 
    print("WARNING: Test input is outside input bounds defined in NNet header!")
    print("Inputs are clipped before evaluation. so evaluations may differ")
    print("Test Input:  "+str(testInput))
    print("Input Mins:  "+str(nnet2.mins))
    print("Input Maxes: "+str(nnet2.maxes))
    print("")
nnetEval2 = nnet2.evaluate_network(testInput)

print("")
print("NNET  Evaluation: "+str(nnetEval))
print("ONNX  Evaluation: "+str(onnxEval))
print("NNET2 Evaluation: "+str(nnetEval2))
print("")
print("Percent Error of ONNX  evaluation: %.8f%%" % (max(abs((nnetEval-onnxEval)/nnetEval))*100.0))
print("Percent Error of NNET2 evaluation: %.8f%%" % (max(abs((nnetEval-nnetEval2)/nnetEval))*100.0))
