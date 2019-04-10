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
    
    
        
    # Open NNet file
    f = open(nnetFile,'r')
    
    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()
        
    # Extract information about network architecture
    record = line.split(',')
    numLayers   = int(record[0])
    inputSize   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers+1,'int')
    for i in range(numLayers+1):
        layerSizes[i]=int(record[i])

    # Skip extra obsolete parameter line
    f.readline()
    
    # Read the normalization information
    line = f.readline()
    inputMins = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    inputMaxes = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",")[:-1]]

    # Initialize list of weights and biases
    weights = [np.zeros((layerSizes[i],layerSizes[i+1])) for i in range(numLayers)]
    biases  = [np.zeros(layerSizes[i+1]) for i in range(numLayers)]

    # Read remainder of file and place each value in the correct spot in a weight matrix or bias vector
    layer=0
    i=0
    j=0
    line = f.readline()
    record = line.split(',')
    while layer+1 < len(layerSizes):
        while i<layerSizes[layer+1]:
            while record[j]!="\n":
                weights[layer][j,i] = float(record[j])
                j+=1
            j=0
            i+=1
            line = f.readline()
            record = line.split(',')

        i=0
        while i<layerSizes[layer+1]:
            biases[layer][i] = float(record[0])
            i+=1
            line = f.readline()
            record = line.split(',')

        layer+=1
        i=0
        j=0
    f.close()
    
    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges
    return weights, biases