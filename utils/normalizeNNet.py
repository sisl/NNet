import numpy as np
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet

def normalizeNNet(readNNetFile, writeNNetFile=None):
    '''
    Normalize the weights and biases of a NNet file.
    
    Args:
        readNNetFile (str): Path to the input .nnet file.
        writeNNetFile (str, optional): Path to the output .nnet file. If None, return the normalized weights and biases.

    Returns:
        weights, biases (optional): If writeNNetFile is None, return the normalized weights and biases.
    '''
    try:
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(readNNetFile, withNorm=True)
        
        numInputs = weights[0].shape[1]
        numOutputs = weights[-1].shape[0]
        
        # Adjust weights and biases of first layer
        for i in range(numInputs):
            weights[0][:, i] /= ranges[i]
        biases[0] -= np.matmul(weights[0], means[:-1])
        
        # Adjust weights and biases of last layer
        weights[-1] *= ranges[-1]
        biases[-1] *= ranges[-1]
        biases[-1] += means[-1]
        
        # Reset means and ranges
        means = np.zeros(numInputs + 1)
        ranges = np.ones(numInputs + 1)
        
        if writeNNetFile:
            writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, writeNNetFile)
        else:
            return weights, biases
    except Exception as e:
        print(f"Error normalizing NNet file: {e}")
        raise

if __name__ == '__main__':
    readNNetFile = '../nnet/TestNetwork.nnet'
    writeNNetFile = '../nnet/TestNetwork3.nnet'
    normalizeNNet(readNNetFile, writeNNetFile)
