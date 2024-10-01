import numpy as np
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet
from typing import Optional, Tuple


def normalizeNNet(readNNetFile: str, writeNNetFile: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize the weights and biases of a neural network read from a .nnet file.

    Args:
        readNNetFile (str): Path to the input .nnet file to read.
        writeNNetFile (Optional[str]): If provided, the normalized network will be saved to this file.
    
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Returns the normalized weights and biases if no write file is provided.
    """
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(readNNetFile, withNorm=True)
    
    numInputs = weights[0].shape[1]
    numOutputs = weights[-1].shape[0]
    
    # Normalize the weights and biases of the first layer
    weights[0] /= ranges[:-1]  # Apply normalization to weights
    biases[0] -= np.matmul(weights[0], means[:-1])  # Adjust biases by subtracting the mean

    # Normalize the weights and biases of the last layer
    weights[-1] *= ranges[-1]  # Apply normalization to output weights
    biases[-1] *= ranges[-1]
    biases[-1] += means[-1]  # Adjust output biases by the output mean

    # Set nominal means and ranges for input/output normalization
    means = np.zeros(numInputs + 1)
    ranges = np.ones(numInputs + 1)

    if writeNNetFile is not None:
        # Write normalized weights and biases to the specified file
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, writeNNetFile)
        return None
    
    # If no file is specified, return the normalized weights and biases
    return weights, biases


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize NNet weights and biases")
    parser.add_argument("readNNetFile", type=str, help="Path to the input .nnet file to read")
    parser.add_argument("--writeNNetFile", type=str, default=None, help="Path to save the normalized .nnet file (optional)")

    args = parser.parse_args()

    # Call the normalization function with the provided arguments
    normalizeNNet(args.readNNetFile, args.writeNNetFile)
