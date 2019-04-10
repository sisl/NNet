import numpy as np
from readNNet import readNNet
from writeNNet import writeNNet

def normalizeNNet(readNNetFile, writeNNetFile):
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(readNNetFile,withNorm=True)
    
    numInputs = weights[0].shape[0]
    numOutputs = weights[-1].shape[1]
    
    # Adjust weights and biases of first layer
    for i in range(numInputs): weights[0][i,:]/=ranges[i]
    biases[0]-= np.matmul(weights[0].T,means[:-1])
    
    # Adjust weights and biases of last layer
    weights[-1]*=ranges[-1]
    biases[-1] *= ranges[-1]
    biases[-1] += means[-1]
    
    # Nominal mean and range vectors
    means = np.zeros(numInputs+1)
    ranges = np.ones(numInputs+1)
    
    writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,writeNNetFile)
   
if __name__=='__main__':
    readNNetFile = '../nnet/TestNetwork.nnet'
    writeNNetFile = '../nnet/TestNetwork3.nnet'
    normalizeNNet(readNNetFile,writeNNetFile)