import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from NNet.utils.writeNNet import writeNNet

# Enable TensorFlow 1.x compatibility in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

def processGraph(op, input_op, foundInputFlag, weights, biases):
    """
    Recursively search the graph and populate the weight and bias lists.
    
    Args:
        op (tf.op): Current tensorflow operation in search.
        input_op (tf.op): Tensorflow operation that we want to be the network input.
        foundInputFlag (bool): Flag turned to true when the input operation is found.
        weights (list): List of weights in network.
        biases (list): List of biases in network.
        
    Returns:
        (bool): Updated foundInputFlag.
    """
    if op.node_def.op == 'Const':
        # If constant, extract values and add to weight or bias list depending on shape
        param = tensor_util.MakeNdarray(op.node_def.attr['value'].tensor)
        if len(param.shape) > 1:
            weights += [param.T]
        else:
            biases += [param]

    # Search the inputs to this operation as well
    input_ops = [i.op for i in op.inputs]
    for i in input_ops:
        # If the operation name is not the given input_op name, recurse. Otherwise, we have found the input operation
        if not i.name == input_op.name:
            foundInputFlag = processGraph(i, input_op, foundInputFlag, weights, biases)
        else:
            foundInputFlag = True
    return foundInputFlag

def pb2nnet(pbFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    """
    Write a .nnet file from a frozen TensorFlow protobuf or SavedModel.

    Args:
        pbFile (str): Path to the frozen graph .pb file or SavedModel folder.
        inputMins (list, optional): Minimum values for each neural network input.
        inputMaxes (list, optional): Maximum values for each neural network output.
        means (list, optional): Mean values for inputs and output for normalization.
        ranges (list, optional): Range values for inputs and output for normalization.
        inputName (str, optional): Name of operation corresponding to input.
        outputName (str, optional): Name of operation corresponding to output.
        savedModel (bool, optional): If True, load SavedModel. If False, load frozen graph. Default is False.
        savedModelTags (list, optional): If loading a SavedModel, specify tags used. Default is an empty list.
    """
    
    if nnetFile == "":
        nnetFile = f"{pbFile[:-2]}nnet"

    if savedModel:
        # Load SavedModel
        sess = tf.compat.v1.Session()
        tf.compat.v1.saved_model.loader.load(sess, savedModelTags, pbFile)
        
        # Simplify graph using outputName, which must be specified for SavedModel
        simp_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [outputName]
        )
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(simp_graph_def, name="")
        sess = tf.compat.v1.Session(graph=graph)
    
    else:
        # Load frozen graph from protobuf
        with tf.compat.v1.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        sess = tf.compat.v1.Session(graph=graph)

    # Find operations corresponding to input and output
    if inputName:
        inputOp = sess.graph.get_operation_by_name(inputName)
    else:
        ops = sess.graph.get_operations()
        placeholders = [x for x in ops if x.node_def.op == 'Placeholder']
        assert len(placeholders) == 1, "Multiple placeholders found, specify inputName."
        inputOp = placeholders[0]

    if outputName:
        outputOp = sess.graph.get_operation_by_name(outputName)
    else:
        outputOp = sess.graph.get_operations()[-1]

    # Recursively search for weights and bias parameters
    weights = []
    biases = []
    foundInputFlag = processGraph(outputOp, inputOp, False, weights, biases)

    inputShape = inputOp.outputs[0].shape.as_list()
    assert inputShape[0] is None
    assert inputShape[1] > 0
    assert len(inputShape) == 2
    inputSize = inputShape[1]

    if foundInputFlag:
        # Default values for input bounds and normalization constants
        if inputMins is None:
            inputMins = inputSize * [np.finfo(np.float32).min]
        if inputMaxes is None:
            inputMaxes = inputSize * [np.finfo(np.float32).max]
        if means is None:
            means = (inputSize + 1) * [0.0]
        if ranges is None:
            ranges = (inputSize + 1) * [1.0]

        # Write NNet file
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
    else:
        print(f"Could not find the given input in graph: {inputOp.name}")

if __name__ == '__main__':
    # Read user inputs and run pb2nnet function
    if len(sys.argv) > 1:
        print("WARNING: Using default values for input bounds and normalization constants")
        pbFile = sys.argv[1]
        pb2nnet(pbFile)
    else:
        print("Need to specify which TensorFlow .pb file to convert to .nnet!")
