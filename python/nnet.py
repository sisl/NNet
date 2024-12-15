import numpy as np

class NNet:
    """
    Class that represents a fully connected ReLU network from a .nnet file.
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
                currentLayerSize = layerSizes[layernum + 1]
                
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
        Evaluate network using given inputs.
        """
        inputs = np.asarray(inputs).flatten()  # Ensure the input is flattened
        if len(inputs) != self.inputSize:
            raise ValueError(f"Expected input size {self.inputSize}, but got {len(inputs)}.")

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
        Evaluate network using multiple sets of inputs.
        """
        inputs = np.asarray(inputs)
        if inputs.ndim != 2 or inputs.shape[1] != self.inputSize:
            raise ValueError(f"Expected input shape (N, {self.inputSize}), but got {inputs.shape}.")

        # Normalize inputs
        inputsNorm = np.array([
            [(self.mins[i] if inputs[j, i] < self.mins[i] else self.maxes[i] if inputs[j, i] > self.maxes[i]
              else inputs[j, i] - self.means[i]) / self.ranges[i]
             for i in range(self.inputSize)]
            for j in range(inputs.shape[0])
        ])

        # Forward pass through the network
        for layer in range(self.numLayers - 1):
            inputsNorm = np.maximum(np.dot(self.weights[layer], inputsNorm.T) + self.biases[layer].reshape(-1, 1), 0)
        outputs = np.dot(self.weights[-1], inputsNorm) + self.biases[-1].reshape(-1, 1)

        # Undo output normalization
        outputs = outputs * self.ranges[-1] + self.means[-1]
        return outputs.T

    def num_inputs(self):
        """
        Get network input size.
        """
        return self.inputSize

    def num_outputs(self):
        """
        Get network output size.
        """
        return self.outputSize
