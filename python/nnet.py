import numpy as np

class NNet:
    """
    Class that represents a fully connected ReLU network from a .nnet file

    Args:
        filename (str): A .nnet file to load

    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        weights (list of numpy arrays): Weight matrices in network
        biases (list of numpy arrays): Bias vectors in network
    """
    
    def __init__(self, filename):
        # Load network from the .nnet file
        with open(filename) as f:
            # Skip the comment lines starting with "//"
            line = f.readline()
            while line.startswith("//"):
                line = f.readline()

            # Read network metadata
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            layerSizes = [int(x) for x in f.readline().strip().split(",")[:-1]]
            symmetric = int(f.readline().strip().split(",")[0])

            # Read input normalization data
            inputMinimums = [float(x) for x in f.readline().strip().split(",")[:-1]]
            inputMaximums = [float(x) for x in f.readline().strip().split(",")[:-1]]
            inputMeans = [float(x) for x in f.readline().strip().split(",")[:-1]]
            inputRanges = [float(x) for x in f.readline().strip().split(",")[:-1]]

            # Read weights and biases for each layer
            weights = []
            biases = []
            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                # Read weights
                weight_matrix = np.zeros((currentLayerSize, previousLayerSize))
                for i in range(currentLayerSize):
                    weight_matrix[i] = [float(x) for x in f.readline().strip().split(",")[:-1]]
                weights.append(weight_matrix)

                # Read biases
                bias_vector = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    bias_vector[i] = float(f.readline().strip().split(",")[0])
                biases.append(bias_vector)

            # Store network parameters
            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges
            self.weights = weights
            self.biases = biases

    def evaluate_network(self, inputs):
        """
        Evaluate network using given inputs

        Args:
            inputs (numpy array of floats): Network inputs to be evaluated
            
        Returns:
            (numpy array of floats): Network output
        """
        # Normalize inputs
        inputsNorm = np.array([
            (self.mins[i] if inputs[i] < self.mins[i] else self.maxes[i] if inputs[i] > self.maxes[i]
             else inputs[i] - self.means[i]) / self.ranges[i]
            for i in range(self.inputSize)
        ])

        # Forward pass through the network
        for layer in range(self.numLayers - 1):
            inputsNorm = np.maximum(np.dot(self.weights[layer], inputsNorm) + self.biases[layer], 0)
        outputs = np.dot(self.weights[-1], inputsNorm) + self.biases[-1]

        # Undo output normalization
        outputs = outputs * self.ranges[-1] + self.means[-1]
        return outputs

    def evaluate_network_multiple(self, inputs):
        """
        Evaluate network using multiple sets of inputs

        Args:
            inputs (numpy array of floats): Array of network inputs to be evaluated.
            
        Returns:
            (numpy array of floats): Network outputs for each set of inputs
        """
        inputs = np.array(inputs).T
        numInputs = inputs.shape[1]

        # Normalize inputs
        inputsNorm = np.array([
            [(self.mins[i] if inputs[i, j] < self.mins[i] else self.maxes[i] if inputs[i, j] > self.maxes[i]
              else inputs[i, j] - self.means[i]) / self.ranges[i]
             for j in range(numInputs)]
            for i in range(self.inputSize)
        ])

        # Forward pass through the network
        for layer in range(self.numLayers - 1):
            inputsNorm = np.maximum(np.dot(self.weights[layer], inputsNorm) + self.biases[layer].reshape(-1, 1), 0)
        outputs = np.dot(self.weights[-1], inputsNorm) + self.biases[-1].reshape(-1, 1)

        # Undo output normalization
        outputs = outputs * self.ranges[-1] + self.means[-1]
        return outputs.T

    def num_inputs(self):
        """
        Get network input size
        """
        return self.inputSize

    def num_outputs(self):
        """
        Get network output size
        """
        return self.outputSize
