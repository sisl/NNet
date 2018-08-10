from writeNNet import writeNNet

#################
# This file shows an example of how 
# These variables need to be edited for specific networks
#################
# Keys in params for the weights and biases
keysW = ['W1','W2','W3','W4','W5','W6','W7']
keysb = ['b1','b2','b3','b4','b5','b6','b7']

# Min and max values used to bound the inputs
inputMins  = [499.0,-3.141593,-3.141593,100.0,0.0]
inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
means  = [1.9485976744e+04,0.0,0.0,468.777778,420.0,7.5188840201005975]
ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# Filename to write .nnet file
fileName = 'out.nnet'
#################


#################
# If using Keras:
#################


######################
# If using Tensorflow:
######################


# Call writeNNet method
writeNNet(params,keysW,keysb,inputMins,inputMaxes,means,ranges,fileName)
