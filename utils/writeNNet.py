import numpy as np
from typing import List

def writeNNet(weights: List[np.ndarray], biases: List[np.ndarray], 
              inputMins: List[float], inputMaxes: List[float], 
              means: List[float], ranges: List[float], 
              fileName: str):
    """
    Write network data to the .nnet file format.

    Args:
        weights (List[np.ndarray]): Weight matrices in network order.
        biases (List[np.ndarray]): Bias vectors in network order.
        inputMins (List[float]): Minimum values for each input.
        inputMaxes (List[float]): Maximum values for each input.
        means (List[float]): Mean values for each input and a mean value for all outputs.
                             Used to normalize inputs/outputs.
        ranges (List[float]): Range values for each input and a range value for all outputs.
                              Used to normalize inputs/outputs.
        fileName (str): Name of the file where the network will be written.
    """

    # Calculate necessary network dimensions
    numLayers = len(weights)
    inputSize = weights[0].shape[1]
    outputSize = biases[-1].shape[0]
    maxLayerSize = max(inputSize, *[b.shape[0] for b in biases])

    # Open the file and write the header and network structure
    with open(fileName, 'w') as f:
        # Write the header
        f.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")
        f.write(f"{numLayers},{inputSize},{outputSize},{maxLayerSize},\n")
        f.write(f"{inputSize}," + ",".join(str(len(b)) for b in biases) + ",\n")
        f.write("0,\n")  # Unused flag
        
        # Write normalization data
        f.write(",".join(map(str, inputMins)) + ",\n")
        f.write(",".join(map(str, inputMaxes)) + ",\n")
        f.write(",".join(map(str, means)) + ",\n")
        f.write(",".join(map(str, ranges)) + ",\n")

        # Write weights and biases layer by layer
        for w, b in zip(weights, biases):
            # Write weights for the current layer
            for row in w:
                f.write(",".join(f"{value:.5e}" for value in row) + ",\n")
            
            # Write biases for the current layer
            for bias_value in b:
                f.write(f"{bias_value:.5e},\n")
