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
    inputMins = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    inputMaxes = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",") if x]

    # Read weights and biases
    weights=[]
    biases = []
    for layernum in range(numLayers):

        previousLayerSize = layerSizes[layernum]
        currentLayerSize = layerSizes[layernum+1]
        weights.append([])
        biases.append([])
        weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
        for i in range(currentLayerSize):
            line=f.readline()
            aux = [float(x) for x in line.strip().split(",")[:-1]]
            for j in range(previousLayerSize):
                weights[layernum][i,j] = aux[j]
        #biases
        biases[layernum] = np.zeros(currentLayerSize)
        for i in range(currentLayerSize):
            line=f.readline()
            x = float(line.strip().split(",")[0])
            biases[layernum][i] = x

    f.close()
    
    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges
    return weights, biases