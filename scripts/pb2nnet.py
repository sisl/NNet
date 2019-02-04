import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from NNet.scripts.writeNNet import writeNNet

def processGraph(op,input_op, foundInputFlag, weights, biases):
    '''
    Recursively search the graph and add popular the weight and bias lists
    
    Args:
        op (tf.op): Current tensorflow operation in search
        input_op (tf.op): Tensorflow operation that we want to be the network input
        foundInputFlag (bool): Flag turned to true when the input operation is found
        weights (list): List of weights in network
        biases (list): List of biases in network
        
    Returns:
        (bool): Updated foundInputFlag
    '''
    
    if op.node_def.op=='Const' and op.outputs[0].consumers()[0].type == 'Identity' :
        # If constant, extract values and add to weight or bias list depending on shape
        param = tensor_util.MakeNdarray(op.node_def.attr['value'].tensor)
        if len(param.shape)>1:
            weights+=[param]
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
        
def pb2nnet(pbFile, inputMins, inputMaxes, means, ranges, order, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    '''
    Constructs a MarabouNetworkTF object from a frozen Tensorflow protobuf or SavedModel
    
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

    sess = pb2sess(pbFile,inputName="", outputName="", savedModel=False, savedModelTags=[])

    FFTF2nnet(sess, inputMins, inputMaxes, means, ranges, order, nnetFile, inputName, outputName)

def pb2sess(pbFile,inputName="", outputName="", savedModel=False, savedModelTags=[]):
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
    return sess

def FFTF2nnet(sess, inputMins, inputMaxes, means, ranges, order, nnetFile="", inputName="", outputName=""):
    import pdb; pdb.set_trace()
    weights, biases = FFTF2W(sess, inputName, outputName)
    writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,order, nnetFile)

def FFTF2W(sess, inputName="", outputName=""):
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
    if foundInputFlag:
        return weights, biases
    else:
        print("Could not find the given input in graph: %s"%inputOp.name)


def pb2W(pbFile, inputName="", outputName="", savedModel=False, savedModelTags=[]):
    sess = pb2sess(pbFile,inputName, outputName, savedModel, savedModelTags)
    weights, biases = FFTF2W(sess, inputName, outputName)
    return weights, biases

def test():
    ## Script showing how to run pb2nnet
    # Min and max values used to bound the inputs
    inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
    inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

    # Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
    means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
    ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

    # Tensorflow pb file to convert to .nnet file
    pbFile = '../nnet/TestNetwork.pb'

    # Convert the file
    pb2nnet(pbFile, inputMins, inputMaxes, means, ranges, order="xW")
