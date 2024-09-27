import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework import graph_util
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

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

    if nnetFile == "":
        nnetFile = pbFile[:-2] + 'nnet'

    if savedModel:
        ### Read SavedModel ###
        sess = tf.compat.v1.Session()
        tf.compat.v1.saved_model.loader.load(sess, savedModelTags, pbFile)

        ### Simplify graph using outputName, which must be specified for SavedModel ###
        simp_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [outputName])
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(simp_graph_def, name="")
        sess = tf.compat.v1.Session(graph=graph)
        ### End reading SavedModel ###

    else:
        ### Read protobuf file and begin session ###
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        sess = tf.compat.v1.Session(graph=graph)

    ### Retrieve input and output tensors ###
    inputTensor = sess.graph.get_tensor_by_name(inputName + ":0")
    outputTensor = sess.graph.get_tensor_by_name(outputName + ":0")

    # Evaluate the graph and save the neural network file
    # Additional logic to convert the graph to NNET format goes here
    
    # Close session
    sess.close()

if __name__ == '__main__':
    # Read user inputs and run pb2nnet function
    if len(sys.argv) > 1:
        pbFile = sys.argv[1]
        nnetFile = ""
        inputName = ""
        outputName = ""
        savedModel = False
        savedModelTags = []

        if len(sys.argv) > 2:
            nnetFile = sys.argv[2]
        if len(sys.argv) > 3:
            inputName = sys.argv[3]
        if len(sys.argv) > 4:
            outputName = sys.argv[4]
        if len(sys.argv) > 5:
            savedModel = bool(sys.argv[5])
        if len(sys.argv) > 6:
            savedModelTags = sys.argv[6:]

        pb2nnet(pbFile, nnetFile=nnetFile, inputName=inputName, outputName=outputName, savedModel=savedModel, savedModelTags=savedModelTags)
    else:
        print("Need to specify a protobuf (.pb) file to convert to NNET format!")
