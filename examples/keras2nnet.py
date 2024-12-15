import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from NNet.utils.writeNNet import writeNNet


# Load Keras network saved using model.save()
keras_file = "KerasTestNetwork.h5"

# The example network has a custom loss function, so it needs to be defined for this example
def asymMSE(y_true, y_pred):
    d = y_true - y_pred                                 # Get prediction errors
    mins = tf.math.argmin(y_true, axis=1)               # Get indices of optimal action
    mins_onehot = tf.one_hot(mins, 5)                   # Convert min index to one hot array 
    others_onehot = mins_onehot - 1                     # Get the indices of the non-optimal actions, which will be -1's
    d_opt = d * mins_onehot                             # Get the error of the optimal action
    d_sub = d * others_onehot                           # Get the errors of the non-optimal actions
    a = 160 * tf.square(d_opt)                          # 160 times error of optimal action squared
    b = tf.square(d_opt)                                # 1   times error of optimal action squared
    c = 40 * tf.square(d_sub)                           # 40  times error of suboptimal actions squared
    d = tf.square(d_sub)                                # 1   times error of suboptimal actions squared
    l = tf.where(d_sub < 0, c, d) + tf.where(d_opt < 0, a, b)  # Apply different penalties depending on sign of errors
    return l

# Load the Keras model, with custom loss function
model = load_model(keras_file, custom_objects={'asymMSE': asymMSE})

# Get a list of the model weights
model_params = model.get_weights()

# Split the network parameters into weights and biases, assuming they alternate
weights = model_params[0:len(model_params):2]
biases = model_params[1:len(model_params):2]

# Transpose weight matrices
weights = [w.T for w in weights]

## Script showing how to run pb2nnet
# Min and max values used to bound the inputs
input_mins  = [0.0, -3.141593, -3.141593, 100.0, 0.0]
input_maxes = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
means  = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
ranges = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0, 373.94992]

# Define the output file for the .nnet format
nnet_file = keras_file[:-2] + 'nnet'

# Convert the file using writeNNet
writeNNet(weights, biases, input_mins, input_maxes, means, ranges, nnet_file)
