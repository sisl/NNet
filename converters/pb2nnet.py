import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from NNet.utils.writeNNet import writeNNet

def processGraph(op,input_op, foundInputFlag, weights, biases):
    '''
    Recursively search the graph and populate the weight and bias lists
    
    Args:
        op (tf.op): Current tensorflow operation in search
        input_op (tf.op): Tensorflow operation that we want to be the network input
        foundInputFlag (bool): Flag turned to true when the input operation is found
        weights (list): List of weights in network
        biases (list): List of biases in network
        
    Returns:
        (bool): Updated foundInputFlag
    '''
    
    if op.node_def.op=='Const':
        # If constant, extract values and add to weight or bias list depending on shape
        param = tensor_util.MakeNdarray(op.node_def.attr['value'].tensor)
        if len(param.shape)>1:
            weights+=[param.T]
        else:
            biases+=[param]
            
    # Search the inputs to this operation as well
    input_ops = [i.op for i in op.inputs]
    for i in input_ops:
        
        # If the operation name is not the given input_op name, recurse. 
        # Otherwise, we have found the input operation
        if not i.name == input_op.name:
            foundInputFlag = processGraph(i, input_op, foundInputFlag, weights, biases)
        else:
            foundInputFlag = True
    return foundInputFlag

        
def pb2nnet(pbFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    '''
    Write a .nnet file from a frozen Tensorflow protobuf or SavedModel

    Args:
        pbFile (str): If savedModel is false, path to the frozen graph .pb file.
                      If savedModel is true, path to SavedModel folder, which
                      contains .pb file and variables subdirectory.
        inputMins (list): Minimum values for each neural network input.
        inputMaxes (list): Maximum values for each neural network output.
        means (list): Mean value for each input and value for mean of all outputs, used for normalization
        ranges (list): Range value for each input and value for range of all outputs, used for normalization
        inputName (str, optional): Name of operation corresponding to input. Default: ""
        outputName (str, optional) Name of operation corresponding to output. Default: ""
        savedModel (bool, optional) If false, load frozen graph. If true, load SavedModel object. Default: False
        savedModelTags (list, optional) If loading a SavedModel, the user must specify tags used. Default: []
    '''
    
    if nnetFile=="":
        nnetFile = pbFile[:-2] + 'nnet'

    if savedModel:
        ### Read SavedModel ###
        sess = tf.Session()
        tf.saved_model.loader.load(sess, savedModelTags, pbFile)

        ### Simplify graph using outputName, which must be specified for SavedModel ###
        simp_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),[outputName])  
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(simp_graph_def, name="")
        sess = tf.Session(graph=graph)
        ### End reading SavedModel

    else:
        ### Read protobuf file and begin session ###
        with tf.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
        ### END reading protobuf ###

    ### Find operations corresponding to input and output ###
    if inputName:
        inputOp = sess.graph.get_operation_by_name(inputName)
    else: # If there is just one placeholder, use it as input
        ops = sess.graph.get_operations()  
        placeholders = [x for x in ops if x.node_def.op == 'Placeholder']
        assert len(placeholders)==1
        inputOp = placeholders[0]
    if outputName:
        outputOp = sess.graph.get_operation_by_name(outputName)
    else: # Assume that the last operation is the output
        outputOp = sess.graph.get_operations()[-1]
    
    # Recursively search for weights and bias parameters and add them to list
    # Search until the inputOp is found
    # If inputOp is not found, than the operation does not exist in the graph or does not lead to the output operation
    weights = []
    biases = []
    foundInputFlag = False
    foundInputFlag = processGraph(outputOp, inputOp, foundInputFlag, weights, biases)
    inputShape = inputOp.outputs[0].shape.as_list()
    assert(inputShape[0] is None)
    assert(inputShape[1] > 0)
    assert(len(inputShape)==2)
    inputSize = inputShape[1]
    if foundInputFlag:
        
        # Default values for input bounds and normalization constants
        if inputMins is None: inputMins = inputSize*[np.finfo(np.float32).min]
        if inputMaxes is None: inputMaxes = inputSize*[np.finfo(np.float32).max]
        if means is None: means = (inputSize+1)*[0.0]
        if ranges is None: ranges = (inputSize+1)*[1.0]
            
        # Write NNet file
        writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,nnetFile)
    else:
        print("Could not find the given input in graph: %s"%inputOp.name)

if __name__ == '__main__':
    # Read user inputs and run pb2nnet function
    # If non-default values of input bounds and normalization constants are needed, this function should be run from a script
    # instead of the command line
    if len(sys.argv)>1:
        print("WARNING: Using the default values of input bounds and normalization constants")
        pbFile = sys.argv[1]
        pb2nnet(pbFile)
    else:
        print("Need to specify which Tensorflow .pb file to convert to .nnet!")
