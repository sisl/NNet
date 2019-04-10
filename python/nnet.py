import numpy as np

class NNet():
    """
    Class that represents NNet
    """
    def __init__ (self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

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
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # Prepare the inputs to the neural network
        inputsNorm = np.zeros(inputSize)
        for i in range(inputSize):
            if inputs[i]<self.mins[i]:
                inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
            elif inputs[i]>self.maxes[i]:
                inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i] 
            else:
                inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i] 

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer],0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1]

        # Undo output normalization
        for i in range(outputSize):
            outputs[i] = outputs[i]*self.ranges[-1]+self.means[-1]
        return outputs

    # Evaluates multiple inputs at once. Each set of inputs should be list of input lists, 
    # or a 2D numpy array with sets of inputs as rows
    # Returns a column of output values for each input set
    def evaluate_network_multiple(self,inputs):
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights
        inputs = np.array(inputs).T

        # Prepare the inputs to the neural network
        numInputs = inputs.shape[1]
        inputsNorm = np.zeros((inputSize,numInputs))
        for i in range(inputSize):
            for j in range(numInputs):
                if inputs[i,j]<self.mins[i]:
                    inputsNorm[i,j] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i,j] > self.maxes[i]:
                    inputsNorm[i,j] = (self.maxes[i]-self.means[i])/self.ranges[i] 
                else:
                    inputsNorm[i,j] = (inputs[i,j]-self.means[i])/self.ranges[i]

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer].reshape((len(biases[layer]),1)),0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1].reshape((len(biases[-1]),1))

        # Undo output normalization
        for i in range(outputSize):
            for j in range(numInputs):
                outputs[i,j] = outputs[i,j]*self.ranges[-1]+self.means[-1]
        return outputs.T

    def num_inputs(self):
        return self.inputSize

    def num_outputs(self):
        return self.outputSize