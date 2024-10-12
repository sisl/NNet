import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import convert_to_constants as ctoc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from NNet.utils.writeNNet import writeNNet

# Enable TensorFlow 1.x functionality in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

def processGraph(op, input_op, foundInputFlag, weights, biases):
    """
    Recursively search the graph and populate the weight and bias lists.
    
    Args:
        op (tf.Operation): Current TensorFlow operation in search.
        input_op (tf.Operation): TensorFlow operation that we want to be the network input.
        foundInputFlag (bool): Flag that is set to True when the input operation is found.
        weights (list): List of weights in the network.
        biases (list): List of biases in the network.
        
    Returns:
        bool: Updated foundInputFlag.
    """
    if op.node_def.op == 'Const':
        # If constant, extract values and add to weight or bias list depending on shape
        param = tensor_util.MakeNdarray(op.node_def.attr['value'].tensor)
        if len(param.shape) > 1:
            weights.append(param.T)  # Transpose the weights for compatibility
        else:
            biases.append(param)
    
    # Search the inputs to this operation
    input_ops = [i.op for i in op.inputs]
    for i in input_ops:
        if i.name != input_op.name:
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
        inputMaxes (list, optional): Maximum values for each neural network input.
        means (list, optional): Mean value for each input and output (for normalization).
        ranges (list, optional): Range value for each input and output (for normalization).
        nnetFile (str, optional): Name of the output .nnet file. Defaults to the name of the .pb file.
        inputName (str, optional): Name of the input operation.
        outputName (str, optional): Name of the output operation.
        savedModel (bool, optional): If True, load SavedModel. If False, load a frozen graph. Default is False.
        savedModelTags (list, optional): Tags to use when loading a SavedModel.
    """
    if not nnetFile:
        nnetFile = f"{pbFile[:-3]}.nnet"

    if savedModel:
        # Load SavedModel
        sess = tf.compat.v1.Session()
        try:
            tf.compat.v1.saved_model.loader.load(sess, savedModelTags, pbFile)
        except Exception as e:
            print(f"Error loading SavedModel: {e}")
            return

        # Simplify the graph using TensorFlow 2.x's convert_variables_to_constants_v2
        simp_graph_def = ctoc.convert_variables_to_constants_v2(sess.graph.as_graph_def(), [outputName])
        with tf.compat.v1.Graph().as_default() as graph:
            tf.import_graph_def(simp_graph_def, name="")
        sess = tf.compat.v1.Session(graph=graph)
    else:
        # Load frozen graph from protobuf
        try:
            with tf.compat.v1.gfile.GFile(pbFile, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.compat.v1.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="")
            sess = tf.compat.v1.Session(graph=graph)
        except Exception as e:
            print(f"Error loading .pb file: {e}")
            return

    # Find operations corresponding to input and output
    try:
        inputOp = sess.graph.get_operation_by_name(inputName) if inputName else [
            x for x in sess.graph.get_operations() if x.node_def.op == 'Placeholder'][0]
        outputOp = sess.graph.get_operation_by_name(outputName) if outputName else sess.graph.get_operations()[-1]
    except Exception as e:
        print(f"Error finding input or output operations: {e}")
        return

    # Recursively search for weights and biases
    weights, biases = [], []
    foundInputFlag = processGraph(outputOp, inputOp, False, weights, biases)
    
    inputShape = inputOp.outputs[0].shape.as_list()
    assert inputShape[0] is None
    assert inputShape[1] > 0
    inputSize = inputShape[1]

    if foundInputFlag:
        # Default values for input bounds and normalization constants
        if inputMins is None:
            inputMins = [np.finfo(np.float32).min] * inputSize
        if inputMaxes is None:
            inputMaxes = [np.finfo(np.float32).max] * inputSize
        if means is None:
            means = [0.0] * (inputSize + 1)
        if ranges is None:
            ranges = [1.0] * (inputSize + 1)

        # Write NNet file
        try:
            writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
            print(f"Converted TensorFlow model at {pbFile} to NNet model at {nnetFile}")
        except Exception as e:
            print(f"Error writing NNet file: {e}")
    else:
        print(f"Could not find the input operation in the graph: {inputOp.name}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("WARNING: Using the default values of input bounds and normalization constants")
        pbFile = sys.argv[1]
        pb2nnet(pbFile)
    else:
        print("Need to specify which TensorFlow .pb file to convert to .nnet!")
