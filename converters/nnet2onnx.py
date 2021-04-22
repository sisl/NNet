import numpy as np 
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2onnx(nnetFile, onnxFile="", outputVar = "y_out", inputVar="X", normalizeNetwork=False):
    '''
    Convert a .nnet file to onnx format
    Args:
        nnetFile: (string) .nnet file to convert to onnx
        onnxFile: (string) Optional, name for the created .onnx file
        outputName: (string) Optional, name of the output variable in onnx
        normalizeNetwork: (bool) If true, adapt the network weights and biases so that 
                                 networks and inputs do not need to be normalized. Default is False.
    '''
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)
        
    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)
    
    # Default onnx filename if none specified
    if onnxFile=="":
        onnxFile = nnetFile[:-4]+'onnx'
    
    # Initialize graph
    inputs = [helper.make_tensor_value_info(inputVar,   TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []
    
    # Loop through each layer of the network and add operations and initializers
    for i in range(numLayers):
        
        # Use outputVar for the last layer
        outputName = "H%d"%i
        if i==numLayers-1: 
            outputName = outputVar
           
        # Weight matrix multiplication
        operations.append(helper.make_node("MatMul",["W%d"%i,inputVar],["M%d"%i]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32),name="W%d"%i))
            
        # Bias add 
        operations.append(helper.make_node("Add",["M%d"%i,"B%d"%i],[outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32),name="B%d"%i))
            
        # Use Relu activation for all layers except the last layer
        if i<numLayers-1: 
            operations.append(helper.make_node("Relu",["H%d"%i],["R%d"%i]))
            inputVar = "R%d"%i
    
    # Create the graph and model in onnx
    graph_proto = helper.make_graph(operations,"nnet2onnx_Model",inputs, outputs,initializers)
    model_def = helper.make_model(graph_proto)

    # Print statements
    print("Converted NNet model at %s"%nnetFile)
    print("    to an ONNX model at %s"%onnxFile)
    
    # Additional print statements if desired
    #print("\nReadable GraphProto:\n")
    #print(helper.printable_graph(graph_proto))
    
    # Save the ONNX model
    onnx.save(model_def, onnxFile)
  

if __name__ == '__main__':
    # Read user inputs and run nnet2onnx function for different numbers of inputs
    if len(sys.argv)>1:
        nnetFile = sys.argv[1]
        if len(sys.argv)>2:
            onnxFile = sys.argv[2]
            if len(sys.argv)>3:
                outputName = argv[3]
                nnet2onnx(nnetFile,onnxFile,outputName)
            else: nnet2onnx(nnetFile,onnxFile)
        else: nnet2onnx(nnetFile)
    else:
        print("Need to specify which .nnet file to convert to ONNX!")
