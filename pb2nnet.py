import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from writeNNet import writeNNet

def processGraph(op,weightList, biasList):
    if op.node_def.op=='Const':
        param = tensor_util.MakeNdarray(op.node_def.attr['value'].tensor)
        if len(param.shape)>1:
            weightList+=[param]
        else:
            biasList+=[param]
    input_ops = [i.op for i in op.inputs]
    for i in input_ops:
        processGraph(i,weightList,biasList)

        
def pb2nnet(pbFile, inputMins, inputMaxes, means, ranges, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    '''
    Constructs a MarabouNetworkTF object from a frozen Tensorflow protobuf or SavedModel
    Args:
        pbFile: (string) If savedModel is false, path to the frozen graph .pb file.
                           If savedModel is true, path to SavedModel folder, which
                           contains .pb file and variables subdirectory.
        inputMins: (list) Minimum values for each neural network input.
        inputMaxes: (list) Maximum values for each neural network output.
        means: (list) Mean value for each input and value for mean of all outputs, used for normalization
        ranges: (list) Range value for each input and value for range of all outputs, used for normalization
        inputName: (string) optional, name of operation corresponding to input.
        outputName: (string) optional, name of operation corresponding to output.
        savedModel: (bool) If false, load frozen graph. If true, load SavedModel object.
        savedModelTags: (list of strings) If loading a SavedModel, the user must specify tags used.
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
    
    weightList = []
    biasList = []
    processGraph(outputOp,weightList,biasList)
    
    params = {}
    keysW = []
    keysb = []
    for i in range(len(weightList)):
        keyW = "w%d"%i
        keyb = "b%d"%i
        params[keyW] = weightList[i]
        params[keyb] = biasList[i]
        keysW += [keyW]
        keysb += [keyb]
    writeNNet(params,keysW,keysb,inputMins,inputMaxes,means,ranges,nnetFile)
    
## Script showing how to run pb2nnet
# Min and max values used to bound the inputs
inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# Tensorflow pb file to convert to .nnet file
pbFile = 'TestNetwork2.pb'

# Convert the file
pb2nnet(pbFile, inputMins, inputMaxes, means, ranges)