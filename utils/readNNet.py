import numpy as np
from typing import List, Tuple, Union

def readNNet(nnetFile: str, withNorm: bool = False) -> Union[
        Tuple[List[np.ndarray], List[np.ndarray]],
        Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[float], List[float]]
    ]:
    """
    Read a .nnet file and return the list of weight matrices and bias vectors.

    Args:
        nnetFile (str): Path to the .nnet file to read.
        withNorm (bool): If True, return normalization parameters (input min, max, means, ranges).
    
    Returns:
        weights (List of np.ndarray): List of weight matrices for the fully connected network.
        biases (List of np.ndarray): List of bias vectors for the fully connected network.
        (Optional) inputMins, inputMaxes, means, ranges (Lists of floats): Normalization parameters.
    """
    weights = []
    biases = []
    
    with open(nnetFile, 'r') as f:
        # Skip header lines starting with "//"
        line = f.readline()
        while line.startswith("//"):
            line = f.readline()
        
        # Extract information about network architecture
        record = line.strip().split(',')
        numLayers = int(record[0])
        inputSize = int(record[1])
        
        # Get layer sizes
        line = f.readline()
        layerSizes = [int(x) for x in line.strip().split(',')]

        # Skip extra obsolete parameter line
        f.readline()
        
        # Read normalization information
        inputMins = [float(x) for x in f.readline().strip().split(",") if x]
        inputMaxes = [float(x) for x in f.readline().strip().split(",") if x]
        means = [float(x) for x in f.readline().strip().split(",") if x]
        ranges = [float(x) for x in f.readline().strip().split(",") if x]

        # Read weights and biases
        for layernum in range(numLayers):
            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]
            
            # Read weights
            layer_weights = np.zeros((currentLayerSize, previousLayerSize))
            for i in range(currentLayerSize):
                line = f.readline()
                layer_weights[i] = [float(x) for x in line.strip().split(",")[:-1]]  # Ignore the trailing comma

            weights.append(layer_weights)
            
            # Read biases
            layer_biases = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line = f.readline()
                layer_biases[i] = float(line.strip().split(",")[0])
                
            biases.append(layer_biases)

    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges

    return weights, biases
