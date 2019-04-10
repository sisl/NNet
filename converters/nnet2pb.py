import tensorflow as tf
import numpy as np 
import sys
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from NNet.utils.readNNet import readNNet

def nnet2pb(nnetFile, pbFile="", output_node_names = "y_out"):
    '''
    Read a .nnet file and create a frozen Tensorflow graph and save to a .pb file
    
    Inputs:
        nnetFile: (string) .nnet file to convert to Tensorflow format
        pbFile: (string) Optional, name for the created .pb file
        output_node_names: (string) Optional, name of the final operation in the Tensorflow graph
    '''
    weights, biases = readNNet(nnetFile)
    inputSize = weights[0].shape[0]
    
    # Default pb filename if none are specified
    if pbFile=="":
        pbFile = nnetFile[:-4]+'pb'
    
    # Reset tensorflow and load a session using only CPUs
    tf.reset_default_graph()
    sess = tf.Session()

    # Define model and assign values to tensors
    currentTensor = tf.placeholder(tf.float32, [None, inputSize],name='input')
    for i in range(len(weights)):
        W = tf.get_variable("W%d"%i, shape=weights[i].shape)
        b = tf.get_variable("b%d"%i, shape=biases[i].shape)
        
        # Use ReLU for all but last operation, and name last operation to desired name
        if i!=len(weights)-1:
            currentTensor = tf.nn.relu(tf.matmul(currentTensor ,W) + b)
        else:
            currentTensor =  tf.add(tf.matmul(currentTensor ,W), b,name=output_node_names)

        # Assign values to tensors
        sess.run(tf.assign(W,weights[i]))
        sess.run(tf.assign(b,biases[i]))
    
    # Freeze the graph to write the pb file
    freeze_graph(sess,pbFile,output_node_names)
    
def freeze_graph(sess, output_graph_name, output_node_names):
    '''
    Given a session with a graph loaded, save only the variables needed for evaluation to a .pb file
    Inputs:
        sess: Tensorflow session where graph is defined
        output_graph_name: Name of file for writing frozen graph
        output_node_names: Name of the output operation in the graph, comma separates if there are multiple output operations
    '''
    
    input_graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,                        # The session is used to retrieve the weights
        input_graph_def,             # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the useful nodes
    ) 

    # Finally we serialize and dump the output graph to the file
    with tf.gfile.GFile(output_graph_name, "w") as f:
        f.write(output_graph_def.SerializeToString())
  
if __name__=='__main__':
    # Read user inputs and run writePB function
    if len(sys.argv)>1:
        nnetFile = sys.argv[1]
        pbFile = ""
        output_node_names = "y_out"
        if len(sys.argv)>2:
            pbFile = sys.argv[2]
        if len(sys.argv)>3:
            output_node_names = argv[3]
        nnet2pb(nnetFile,pbFile,output_node_names)
    else:
        print("Need to specify which .nnet file to convert to Tensorflow frozen graph!")