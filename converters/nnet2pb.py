import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

tf.compat.v1.disable_eager_execution()

def nnet2pb(nnetFile, pbFile=None, output_node_names="y_out", normalizeNetwork=False):
    """
    Converts a .nnet file to a frozen TensorFlow graph and saves it as a .pb file.
    """
    # Ensure pbFile has a valid default value
    if pbFile is None:
        pbFile = f"{nnetFile[:-5]}.pb"  # Avoids '..pb' filename issue

    # Reset TensorFlow graph and create a session
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    # Read weights and biases from the NNet file
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except Exception as e:
        print(f"Error reading or normalizing NNet file: {e}")
        return

    inputSize = weights[0].shape[1]
    currentTensor = tf.compat.v1.placeholder(tf.float32, [None, inputSize], name='input')

    # Define the network structure using weights and biases
    for i, (W_value, b_value) in enumerate(zip(weights, biases)):
        W = tf.Variable(W_value.T, dtype=tf.float32, name=f"W{i}")
        b = tf.Variable(b_value, dtype=tf.float32, name=f"b{i}")

        if i != len(weights) - 1:
            currentTensor = tf.nn.relu(tf.matmul(currentTensor, W) + b)
        else:
            currentTensor = tf.add(tf.matmul(currentTensor, W), b, name=output_node_names)

    sess.run(tf.compat.v1.global_variables_initializer())

    # Freeze the graph and save the .pb file
    try:
        freeze_graph_v2(sess, pbFile, output_node_names)
        print(f"Successfully saved TensorFlow frozen graph to {pbFile}")
    except Exception as e:
        print(f"Error freezing or saving the graph: {e}")

def freeze_graph_v2(sess, output_graph_name, output_node_names):
    """
    Freezes the TensorFlow session into a .pb file.
    """
    input_graph_def = sess.graph.as_graph_def()

    @tf.function
    def model_function():
        tf.import_graph_def(input_graph_def, name="")

    concrete_function = model_function.get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(concrete_function)

    with tf.io.gfile.GFile(output_graph_name, "wb") as f:
        f.write(frozen_func.graph.as_graph_def().SerializeToString())

if __name__ == '__main__':
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        pbFile = sys.argv[2] if len(sys.argv) > 2 else None
        output_node_names = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2pb(nnetFile, pbFile, output_node_names)
    else:
        print("Error: Need to specify which .nnet file to convert to TensorFlow frozen graph!")
