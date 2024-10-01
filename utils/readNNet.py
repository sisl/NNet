import numpy as np

def readNNet(nnetFile, withNorm=False):
    '''
    Read a .nnet file and return list of weight matrices and bias vectors
    
    Inputs:
        nnetFile: (string) .nnet file to read
        withNorm: (bool) If true, return normalization parameters
        
    Returns: 
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''
    try:
        # Open NNet file
        with open(nnetFile, 'r') as f:
            # Skip header lines
            line = f.readline()
            while line[:2] == "//":
                line = f.readline()

            # Extract information about network architecture
            record = line.split(',')
            numLayers = int(record[0])
            inputSize = int(record[1])

            line = f.readline()
            layerSizes = [int(x) for x in line.strip().split(',') if x]

            # Ensure that the architecture information is correct
            assert len(layerSizes) == numLayers + 1, "Layer sizes don't match number of layers."

            # Skip extra obsolete parameter line
            f.readline()

            # Read the normalization information
            inputMins = [float(x) for x in f.readline().strip().split(",") if x]
            inputMaxes = [float(x) for x in f.readline().strip().split(",") if x]
            means = [float(x) for x in f.readline().strip().split(",") if x]
            ranges = [float(x) for x in f.readline().strip().split(",") if x]

            # Read weights and biases
            weights = []
            biases = []
            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]

                weight_matrix = np.zeros((currentLayerSize, previousLayerSize))
                for i in range(currentLayerSize):
                    line = f.readline()
                    weight_matrix[i] = [float(x) for x in line.strip().split(",")[:-1]]
                weights.append(weight_matrix)

                bias_vector = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line = f.readline()
                    bias_vector[i] = float(line.strip().split(",")[0])
                biases.append(bias_vector)

            if withNorm:
                return weights, biases, inputMins, inputMaxes, means, ranges
            return weights, biases
    except Exception as e:
        print(f"Error reading NNet file: {e}")
        raise
