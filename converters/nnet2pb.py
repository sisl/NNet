import tensorflow as tf
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

def nnet2pb(nnetFile, pbFile="", output_node_names="y_out", normalizeNetwork=False):
    '''
    Read a .nnet file and create a frozen TensorFlow graph and save to a .pb file
    
    Args:
        nnetFile (str): A .nnet file to convert to TensorFlow format
        pbFile (str, optional): Name for the created .pb file. Default: ""
        output_node_names (str, optional): Name of the final operation in the TensorFlow graph. Default: "y_out"
    '''
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)
    inputSize = weights[0].shape[1]
    
    # Default pb filename if none are specified
    if pbFile == "":
        pbFile = nnetFile[:-4] + 'pb'
    
    # Define the model with layers in TensorFlow 2.x style
    inputs = tf.keras.Input(shape=(inputSize,), name='input')
    currentTensor = inputs
    for i in range(len(weights)):
        W = tf.constant(weights[i].T, dtype=tf.float32)
        b = tf.constant(biases[i], dtype=tf.float32)
        
        # Use ReLU for all but the last operation, and name last operation to desired name
        if i != len(weights) - 1:
            currentTensor = tf.nn.relu(tf.matmul(currentTensor, W) + b)
        else:
            currentTensor = tf.add(tf.matmul(currentTensor, W), b, name=output_node_names)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=currentTensor)

    # Save the model to a .pb file
    tf.saved_model.save(model, pbFile)

if __name__ == '__main__':
    # Read user inputs and run nnet2pb function
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
        print("Need to specify which .nnet file to convert to TensorFlow frozen graph!")
