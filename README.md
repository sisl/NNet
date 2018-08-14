## NNet Repository

### Introduction
The .nnet file format for fully connected ReLU networks was originially created in 2016 to define aircraft collision avoidance neural networks in a human-readable text document. Since then it was incorporated into the Reluplex repository and used to define benchmark neural networks.

This repository contains documentation for the .nnet format as well as example scripts to read/write .nnet files. The nnet folder contains example neural network files, the scripts folder contains python scripts to generate the .nnet files from Tensorflow/Keras as well as to generate a Tensorflow frozen model from a .nnet file, and the folder folder contains Python, Julia, and C++ examples for reading and evaluating .nnet networks.

### File format of .nnet
The file begins with a header line, some information about the network architecture, normalization information, and then model parameters. Line by line:<br/><br/>
    **1**: Header text. This can be any number of lines so long as they begin with "//"<br/>
    **2**: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size<br/>
    **3**: A sequence of values describing the network layer sizes. Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size<br/>
    **4**: A flag that is no longer used, can be ignored<br/>
    **5**: Minimum values of inputs (used to keep inputs within expected range)<br/>
    **6**: Maximum values of inputs (used to keep inputs within expected range)<br/>
    **7**: Mean values of inputs and one value for all outputs (used for normalization)<br/>
    **8**: Range values of inputs and one value for all outputs (used for normalization)<br/>
    **9+**: Begin defining the weight matrix for the first layer, followed by the bias vector. The weights and biases for the second layer follow after, until the weights and biases for the output layer are defined.<br/>
    
The minimum/maximum input values are used to define the range of input values seen during training, which can be used to ensure input values remain in the training range. Each input has its own value.

The mean/range values are the values used to normalize the network training data before training the network. The normalization substracts the mean and divides by the range, giving a distribution that is zero mean and unit range. Therefore, new inputs to the network should be normalized as well, so there is a mean/range value for every input to the network. There is also an additional mean/range value for the network outputs, but just one value for all outputs. The raw network outputs can be re-scaled by multiplying by the range and adding the mean.

### Writing .nnet files
In the scripts folder, the file writeNNet.py conatins a python method for writing neural network data to a .nnet file. The main method, writeNNet, requires a list of weights, biases, minimum input values, maximum input values, mean of inputs/ouput, and range of inputs/output, and a filename to write the neural network. This function can be imported to any python file using "from writeNNet import writeNNet", assuming that the scripts folder is the current directory or is in the PYTHONPATH.

The pb2nnet.py used the writeNNet function to show how a frozen Tensorflow model or SavedModel can be converted to the .nnet format, assuming the model is a linear, fully connected ReLU model. Also, keras2nnet.py demonstrates how the weights can be extracted from a Keras model and used to write a .nnet file. Lastly, the .nnet file can be used to create a frozen Tensorflow model using nnet2pb.py. Converting a .nnet file to a frozen Tensorflow file and then back again shows that the file format conversions do not change the model.

### Loading and evaluating .nnet files
In the readNNet folder, there are three subfolders for C++, Julia, and Python examples. Each subfolder contains a nnet.* file that contains functions for loading the network from a .nnet file and then evaluating a set of inputs given the loaded model. There are examples in each folder to demonstrate how the functions can be used.

## License
This code is licensed under the MIT license. See LICENSE for details.
