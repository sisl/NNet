import numpy as np
from tensorflow.python.framework import tensor_util, graph_util
import tensorflow as tf
import os
from NNet.utils.writeNNet import writeNNet

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

def processGraph(op, input_op, foundInputFlag, weights, biases):
    '''
    Recursively search the graph to populate weights and biases lists.

    Args:
        op (tf.Operation): Current TensorFlow operation in search.
        input_op (tf.Operation): Operation that corresponds to the network input.
        foundInputFlag (bool): True if input operation is found.
        weights (list): List to store weight matrices.
        biases (list): List to store bias vectors.

    Returns:
        bool: Updated foundInputFlag.
    '''
    if op.type == 'Const':
        param = tensor_util.MakeNdarray(op.get_attr('value'))
        if len(param.shape) > 1:
            weights.append(param.T)  # Transpose to match .nnet format
        else:
            biases.append(param)

    input_ops = [i.op for i in op.inputs]
    for i in input_ops:
        if i.name != input_op.name:
            foundInputFlag = processGraph(i, input_op, foundInputFlag, weights, biases)
        else:
            foundInputFlag = True
    return foundInputFlag

def pb2nnet(pbFile, inputMins=None, inputMaxes=None, means=None, ranges=None, 
            nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
    '''
    Convert a TensorFlow protobuf or SavedModel to a .nnet file.

    Args:
        pbFile (str): Path to the frozen graph or SavedModel folder.
        inputMins (list): Min values for each input.
        inputMaxes (list): Max values for each input.
        means (list): Mean values for normalization.
        ranges (list): Range values for normalization.
        nnetFile (str): Output .nnet file name. Default: "".
        inputName (str): Name of input operation. Default: "".
        outputName (str): Name of output operation. Default: "".
        savedModel (bool): Set True for SavedModel format. Default: False.
        savedModelTags (list): Tags to load SavedModel. Default: [].
    '''
    if nnetFile == "":
        nnetFile = pbFile.replace(".pb", ".nnet")

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    if savedModel:
        tf.compat.v1.saved_model.loader.load(sess, savedModelTags, pbFile)
        graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [outputName]
        )
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
    else:
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

    sess = tf.compat.v1.Session(graph=graph)

    # Identify input and output operations
    if inputName:
        inputOp = sess.graph.get_operation_by_name(inputName)
    else:
        placeholders = [op for op in sess.graph.get_operations() if op.type == 'Placeholder']
        assert len(placeholders) == 1, "Multiple placeholders found, specify inputName."
        inputOp = placeholders[0]

    if outputName:
        outputOp = sess.graph.get_operation_by_name(outputName)
    else:
        outputOp = sess.graph.get_operations()[-1]

    # Recursively find weights and biases
    weights, biases = [], []
    foundInputFlag = processGraph(outputOp, inputOp, False, weights, biases)

    inputShape = inputOp.outputs[0].shape.as_list()
    assert inputShape[0] is None, "Batch size must be None."
    assert len(inputShape) == 2, "Input must be a 2D tensor."
    inputSize = inputShape[1]

    if foundInputFlag:
        inputMins = inputMins or [np.finfo(np.float32).min] * inputSize
        inputMaxes = inputMaxes or [np.finfo(np.float32).max] * inputSize
        means = means or [0.0] * (inputSize + 1)
        ranges = ranges or [1.0] * (inputSize + 1)

        print(f"Converted TensorFlow model '{pbFile}' to NNet file '{nnetFile}'.")
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
    else:
        print(f"Error: Input operation '{inputOp.name}' not found in the graph.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("WARNING: Using default input bounds and normalization constants.")
        pbFile = sys.argv[1]
        pb2nnet(pbFile)
    else:
        print("Usage: python pb2nnet.py <pbFile> [<nnetFile>]")
