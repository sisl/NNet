import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

# Enable TensorFlow 1.x functionalities in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

def nnet2pb(nnetFile, pbFile="", output_node_names="y_out", normalizeNetwork=False):
    """
    Convert a .nnet file into a frozen TensorFlow graph and save it to a .pb file.

    Args:
        nnetFile (str): Path to the .nnet file to convert.
        pbFile (str, optional): Name for the created .pb file. Default: generated from input filename.
        output_node_names (str, optional): Name of the final operation in the TensorFlow graph. Default is "y_out".
        normalizeNetwork (bool, optional): If True, normalize the network. Default is False.
    """
    try:
        if normalizeNetwork:
            weights, biases = normalizeNNet(nnetFile)
        else:
            weights, biases = readNNet(nnetFile)
    except Exception as e:
        print(f"Error reading or normalizing NNet file: {e}")
        return

    inputSize = weights[0].shape[1]

    # Default pb filename if none is specified
    if not pbFile:
        pbFile = f"{nnetFile[:-4]}.pb"

    # Reset TensorFlow graph and initialize session
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    # Define the model structure and assign values to tensors
    currentTensor = tf.compat.v1.placeholder(tf.float32, [None, inputSize], name='input')

    for i, (W_value, b_value) in enumerate(zip(weights, biases)):
        W = tf.compat.v1.get_variable(f"W{i}", shape=W_value.T.shape)
        b = tf.compat.v1.get_variable(f"b{i}", shape=b_value.shape)

        # Apply ReLU activation except for the last layer
        if i != len(weights) - 1:
            currentTensor = tf.nn.relu(tf.matmul(currentTensor, W) + b)
        else:
            currentTensor = tf.add(tf.matmul(currentTensor, W), b, name=output_node_names)

        # Assign values to the variables
        sess.run(tf.compat.v1.assign(W, W_value.T))
        sess.run(tf.compat.v1.assign(b, b_value))

    # Freeze the graph and save the .pb file
    try:
        freeze_graph(sess, pbFile, output_node_names)
        print(f"Successfully saved TensorFlow frozen graph to {pbFile}")
    except Exception as e:
        print(f"Error freezing or saving the graph: {e}")

def freeze_graph(sess, output_graph_name, output_node_names):
    """
    Save only the necessary variables for evaluation to a .pb file.

    Args:
        sess (tf.compat.v1.Session): The TensorFlow session where the graph is defined.
        output_graph_name (str): The name of the file to save the frozen graph.
        output_node_names (str): Name(s) of the output operation(s) in the graph.
    """
    try:
        input_graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names.split(",")
        )

        # Serialize and save the frozen graph
        with tf.io.gfile.GFile(output_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    except Exception as e:
        print(f"Error during graph freezing or file writing: {e}")
        raise

if __name__ == '__main__':
    # Read user inputs and run nnet2pb function
    if len(sys.argv) > 1:
        nnetFile = sys.argv[1]
        pbFile = sys.argv[2] if len(sys.argv) > 2 else ""
        output_node_names = sys.argv[3] if len(sys.argv) > 3 else "y_out"
        nnet2pb(nnetFile, pbFile, output_node_names)
    else:
        print("Error: Need to specify which .nnet file to convert to TensorFlow frozen graph!")
