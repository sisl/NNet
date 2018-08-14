import tensorflow as tf
import numpy as np 
import sys
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def nnet2pb(nnetFile, pbFile="", output_node_names = "y_out"):
    '''
    Read a .nnet file and create a frozen Tensorflow graph and save to a .pb file
    
    Args:
        nnetFile (str): A .nnet file to convert to Tensorflow format
        pbFile (str, optional): Name for the created .pb file. Default: ""
        output_node_names (str, optional): Name of the final operation in the Tensorflow graph. Default: "y_out"
    '''
    
    # Default pb filename if none are specified
    if pbFile=="":
        pbFile = nnetFile[:-4]+'pb'
        
    # Open NNet file
    f = open(nnetFile,'r')
    
    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()
        
    # Extract information about network architecture
    record = line.split(',')
    numLayers   = int(record[0])
    inputSize   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers+1,'int')
    for i in range(numLayers+1):
        layerSizes[i]=int(record[i])

    # Skip the normalization information
    f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()

    # Initialize list of weights and biases
    weights = [np.zeros((layerSizes[i],layerSizes[i+1])) for i in range(numLayers)]
    biases  = [np.zeros(layerSizes[i+1]) for i in range(numLayers)]

    # Read remainder of file and place each value in the correct spot in a weight matrix or bias vector
    layer=0
    i=0
    j=0
    line = f.readline()
    record = line.split(',')
    while layer+1 < len(layerSizes):
        while i<layerSizes[layer+1]:
            while record[j]!="\n":
                weights[layer][j,i] = float(record[j])
                j+=1
            j=0
            i+=1
            line = f.readline()
            record = line.split(',')

        i=0
        while i<layerSizes[layer+1]:
            biases[layer][i] = float(record[0])
            i+=1
            line = f.readline()
            record = line.split(',')

        layer+=1
        i=0
        j=0
    f.close()
    
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
    
    Args:
        sess (tf.session): Tensorflow session where graph is defined
        output_graph_name (str): Name of file for writing frozen graph
        output_node_names (str): Name of the output operation in the graph, comma separated if there are multiple output operations
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
