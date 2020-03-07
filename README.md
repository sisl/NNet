## NNet Repository

[![Build Status](https://travis-ci.org/sisl/NNet.svg?branch=master)](https://travis-ci.org/sisl/NNet)
[![Coverage Status](https://coveralls.io/repos/github/sisl/NNet/badge.svg?branch=master&service=github)](https://coveralls.io/github/sisl/NNet?branch=master)

### Introduction
The .nnet file format for fully connected ReLU networks was originially created in 2016 to define aircraft collision avoidance neural networks in a human-readable text document. Since then it was incorporated into the Reluplex repository and used to define benchmark neural networks. This format is a simple text-based format for feed-forward, fully-connected, ReLU-activated neural networks. It is not affiliated with Neuroph or other frameworks that produce files with the .nnet extension.

This repository contains documentation for the .nnet format as well as useful functions for working with the networks. The nnet folder contains example neural network files. The converters folder contains functions to convert the .nnet files to Tensorflow, ONNX, and Keras formats and vice-versa. The python, julia, and cpp folders contain python, julia, and C++ functions for reading and evaluating .nnet networks. The examples folder provides python examples for using the available functions.

This repository is set up as a python package. To run the examples, make sure that the folder in which this repository resides (the parent directory of NNet) is added to the PYTHONPATH environment variable.

### File format of .nnet
The file begins with header lines, some information about the network architecture, normalization information, and then model parameters. Line by line:<br/><br/>
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
In the utils folder, the file writeNNet.py contains a python method for writing neural network data to a .nnet file. The main method, writeNNet, requires a list of weights, biases, minimum input values, maximum input values, mean of inputs/ouput, and range of inputs/output, and a filename to write the neural network.

### Loading and evaluating .nnet files
There are three folders for C++, Julia, and Python examples. Each subfolder contains a nnet.* file that contains functions for loading the network from a .nnet file and then evaluating a set of inputs given the loaded model. There are examples in each folder to demonstrate how the functions can be used.

## License
This code is licensed under the MIT license. See LICENSE for details.
