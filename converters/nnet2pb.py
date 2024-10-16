import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

def nnet2pb(nnetFile, pbFile="", output_node_names="y_out", normalizeNetwork=False):
    '''
    Convert a .nnet file to a frozen TensorFlow graph and save it as a .pb file.

    Args:
        nnetFile (str): Path to the .nnet file.
        pbFile (str, optional): Name for the output .pb file. Default: ""
        output_node_names (str, optional): Name of the output operation. Default: "y_out"
        normalizeNetwork (bool, optional): If True, normalize the network. Default: False.
    '''
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)

    inputSize = weights[0].shape[1]

    # Default pb filename if not provided
    if pbFile == "":
        pbFile = nnetFile.replace(".nnet", ".pb")

    # Use TensorFlow v1 compatibility mode
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # Define the model
        input_tensor = tf.compat.v1.placeholder(tf.float32, [None, inputSize], name='input')

        current_tensor = input_tensor
        for i in range(len(weights)):
            W = tf.Variable(weights[i].T, dtype=tf.float32, name=f"W{i}")
            b = tf.Variable(biases[i], dtype=tf.float32, name=f"b{i}")

            # Apply MatMul and BiasAdd for each layer
            if i != len(weights) - 1:
                current_tensor = tf.nn.relu(tf.matmul(current_tensor, W) + b)
            else:
                current_tensor = tf.add(tf.matmul(current_tensor, W), b, name=output_node_names)

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Freeze the graph and save to a .pb file
        freeze_graph(sess, pbFile, output_node_names)

def freeze_graph(sess, output_graph_name, output_node_names):
    '''
    Freeze the TensorFlow graph to save only the variables needed for evaluation.

    Args:
        sess (tf.compat.v1.Session): TensorFlow session containing the graph.
        output_graph_name (str): Name of the .pb file to save.
        output_node_names (str): Name of the output operation(s), comma-separated if multiple.
    '''
    input_graph_def = sess.graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(",")
    )

    # Save the frozen graph
    with tf.io.gfile.GFile(output_graph_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print(f"Saved frozen graph to {output_graph_name}")

if __name__ == '__main__':
    # Read command-line arguments
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        pbFile = sys.argv[2] if len(sys.argv) > 2 else ""
        output_node_names = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2pb(nnetFile, pbFile, output_node_names)
    else:
        print("Usage: python nnet2pb.py <nnetFile> [<pbFile>] [<output_node_names>]")
