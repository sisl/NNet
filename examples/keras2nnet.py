import numpy as np
import h5py
from keras.models import load_model
import theano.tensor as T
from NNet.utils.writeNNet import writeNNet


# Load Keras network saved using model.save()
kerasFile = "KerasTestNetwork.h5"

# The example network has a custom loss function, so it needs to be defined for this example
def asymMSE(y_true, y_pred):
    d = y_true-y_pred                              # Get prediction errors
    mins = T.argmin(y_true,axis=1)                 # Get indices of optimal action
    mins_onehot = T.extra_ops.to_one_hot(mins,5)   # Convert min index to one hot array 
    others_onehot = mins_onehot-1                  # Get the indices of the non-optimal actions, which will be -1's
    d_opt = d*mins_onehot                          # Get the error of the optimal action
    d_sub = d*others_onehot                        # Get the errors of the non-optimal actions
    a = 160*d_opt**2                               # 160 times error of optimal action squared
    b = d_opt**2                                   # 1   times error of optimal action squared
    c = 40*d_sub**2                                # 40  times error of suboptimal actions squared
    d = d_sub**2                                   # 1   times error of suboptimal actions squared
    l = T.switch(d_sub<0,c,d) + T.switch(d_opt<0,a,b) #This chooses which errors to use depending on the sign of the errors
                                                      #If true, use the steeper penalty. If false, use the milder penalty
    return l
model = load_model(kerasFile,custom_objects = {'asymMSE':asymMSE})

# Get a list of the model weights
model_params = model.get_weights()

# Split the network parameters into weights and biases, assuming they alternate
weights = model_params[0:len(model_params):2]
biases  = model_params[1:len(model_params):2]

# Transpose weight matrices
weights = [w.T for w in weights]
    
## Script showing how to run pb2nnet
# Min and max values used to bound the inputs
inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# Tensorflow pb file to convert to .nnet file
nnetFile = kerasFile[:-2] + 'nnet'

# Convert the file
writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,nnetFile)
