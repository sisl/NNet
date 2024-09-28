import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

# Enable TensorFlow 1.x functionalities in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

def nnet2pb(nnetFile, pbFile="", output_node_names="y_out", normalizeNetwork=False):
    """
    Read a .nnet file and create a frozen Tensorflow graph and save to a .pb file

    Args:
        nnetFile (str): A .nnet file to convert to Tensorflow format
        pbFile (str, optional): Name for the created .pb file. Default: ""
        output_node_names (str, optional): Name of the final operation in the Tensorflow graph. Default: "y_out"
    """
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)
    
    inputSize = weights[0].shape[1]

    # Default pb filename if none are specified
    if pbFile == "":
        pbFile = nnetFile[:-4] + 'pb'

    # Reset TensorFlow and start a session using only CPUs
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    # Define model and assign values to tensors
    currentTensor = tf.compat.v1.placeholder(tf.float32, [None, inputSize], name='input')

    for i in range(len(weights)):
        W = tf.compat.v1.get_variable(f"W{i}", shape=weights[i].T.shape)
        b = tf.compat.v1.get_variable(f"b{i}", shape=biases[i].shape)

        # Use ReLU for all but the last operation, name the last operation to desired name
        if i != len(weights) - 1:
            currentTensor = tf.nn.relu(tf.matmul(currentTensor, W) + b)
        else:
            currentTensor = tf.add(tf.matmul(currentTensor, W), b, name=output_node_names)

        # Assign values to tensors
        sess.run(tf.compat.v1.assign(W, weights[i].T))
        sess.run(tf.compat.v1.assign(b, biases[i]))

    # Freeze the graph and write the .pb file
    freeze_graph(sess, pbFile, output_node_names)

def freeze_graph(sess, output_graph_name, output_node_names):
    """
    Given a session with a graph loaded, save only the variables needed for evaluation to a .pb file

    Args:
        sess (tf.compat.v1.Session): Tensorflow session where graph is defined
        output_graph_name (str): Name of file for writing frozen graph
        output_node_names (str): Name of the output operation in the graph, comma-separated if there are multiple output operations
    """

    input_graph_def = tf.compat.v1.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,                        # The session is used to retrieve the weights
        input_graph_def,             # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the useful nodes
    )

    # Serialize and dump the output graph to the file
    with tf.io.gfile.GFile(output_graph_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    # Read user inputs and run the nnet2pb function
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        pbFile = ""
        output_node_names = "y_out"
        if len(sys.argv) > 2:
            pbFile = sys.argv[2]
        if len(sys.argv) > 3:
            output_node_names = sys.argv[3]
        nnet2pb(nnetFile, pbFile, output_node_names)
    else:
        print("Need to specify which .nnet file to convert to Tensorflow frozen graph!")
