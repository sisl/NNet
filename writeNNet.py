import numpy as np

def writeNNet(params,keysW,keysb,inputMins,inputMaxes,means,ranges,fileName):
    '''
    Write network data to the .nnet file format
    
    params: dictionary of network variables. This can be extracted from tensorflow with
        params = sess.run(variable_dict)
        where variable_dict is a dictionary of network variables
    keysW: List of keys corresponding to weight matrices in the params dictionary
    keysb: List of keys corresponding to bias vectors in the params dictionary
    inputMins: List of minimum values for each input
    inputMaxes: List of maximum values for each input
    means: List of mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
    ranges: List of range values for each input and a range value for all outputs. Used to normalize inputs/outputs
    fileName: File where the network will be written
    '''

   
    
    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of hidden layers in the networks
        #     Number of inputs to the networks
        #     Number of outputs from the networks
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line specifies if the network is "symmetric", in which the network was only 
        #     trained on half of the state space and then the other half is mirrored. This
        #     option was explored but not fruitfully, so this value is just set to false, 0
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of the outputs
        # The eighth line gives the range of each input and of the outputs
        #     These two lines are used to map raw inputs to the 0 mean, 1 range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(keysW)
        inputSize = params[keysW[0]].shape[0]
        outputSize = params[keysb[-1]].shape[0]
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for key in keysW:
            if params[key].shape[1]>maxLayerSize :
                maxLayerSize = params[key].shape[1]

        # Write data to header 
        str = "%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize)
        f2.write(str)
        str = "%d," % inputSize
        f2.write(str)
        for key in keysW:
            str = "%d," % params[key].shape[1]
            f2.write(str)
        f2.write("\n")
        f2.write("0,\n") #Symmetric Boolean
        
        # Write Min, Max, Mean, and Range of each of the inputs on outputs for normalization
        f2.write(','.join(str(inputMins[i])  for i in range(inputSize)) + '\n') #Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + '\n') #Maximum Input Values                
        f2.write(','.join(str(means[i])      for i in range(inputSize+1)) + '\n') #Means for normalizations
        f2.write(','.join(str(ranges[i])     for i in range(inputSize+1)) + '\n') #Ranges for noramlizations

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for ii in range(len(keysW)):
            data = np.array(params[keysW[ii]]).T
            for i in range(len(data)):
                for j in range(int(np.size(data)/len(data))):
                    str = ""
                    if int(np.size(data)/len(data))==1:
                        str = "%.5e," % data[i] #Five digits written. More can be used, but that requires more more space.
                    else:
                        str = "%.5e," % data[i][j]
                    f2.write(str)
                f2.write("\n")
            data = np.array(params[keysb[ii]]).T
            for i in range(len(data)):
                for j in range(int(np.size(data)/len(data))):
                    str = ""
                    if int(np.size(data)/len(data))==1:
                        str = "%.5e," % data[i] #Five digits written. More can be used, but that requires more more space.
                    else:
                        str = "%.5e," % data[i][j]
                    f2.write(str)
                f2.write("\n")
