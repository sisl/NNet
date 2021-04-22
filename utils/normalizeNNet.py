import numpy as np
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet

def normalizeNNet(readNNetFile, writeNNetFile=None):
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(readNNetFile,withNorm=True)
    
    numInputs = weights[0].shape[1]
    numOutputs = weights[-1].shape[0]
    
    # Adjust weights and biases of first layer
    for i in range(numInputs): weights[0][:,i]/=ranges[i]
    biases[0]-= np.matmul(weights[0],means[:-1])
    
    # Adjust weights and biases of last layer
    weights[-1]*=ranges[-1]
    biases[-1] *= ranges[-1]
    biases[-1] += means[-1]
    
    # Nominal mean and range vectors
    means = np.zeros(numInputs+1)
    ranges = np.ones(numInputs+1)
    
    if writeNNetFile is not None:
        writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,writeNNetFile)
        return None
    return weights, biases
   
if __name__ == '__main__':
    readNNetFile = '../nnet/TestNetwork.nnet'
    writeNNetFile = '../nnet/TestNetwork3.nnet'
    normalizeNNet(readNNetFile,writeNNetFile)
