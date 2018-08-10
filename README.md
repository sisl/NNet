## NNet Repository

### Introduction
The .nnet file format for fully connected ReLU networks was originially created to define aircraft collision avoidance neural networks in a human-readable text document. Since then it was incorporated into the Reluplex repository and an input parser was created to read the .nnet files.

This repository contains documentation on the .nnet format as well as example scripts to read/write .nnet files using Python, Julia, and C++.

### File Format of .nnet
The file begins with a header line, some information about the network architecture, normalization information, and then model parameters. Line by line:<br/><br/>
    **1**: Header text<br/>
    **2**: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size<br/>
    **3**: A sequence of values describing the network layer sizes. Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size<br/>
    **4**: Unused flag, can ignore<br/>
    **5**: Minimum values of inputs (used to keep inputs within expected range)<br/>
    **6**: Maximum values of inputs (used to keep inputs within expected range)<br/>
    **7**: Mean values of inputs (used for normalization)<br/>
    **8**: Range values of inputs (used for normalization)<br/>
    **9+**: Begin defining the weight matrix for the first layer, followed by the bias vector. The weights and biases for the second layer follow after, until the weights and biases for the output layer are defined.<br/>

### Writing .nnet files
The file writeNNet.py conatins a python method for writing neural network data from Keras or Tensorflow to a .nnet file. The main method, writeNNet(), takes a dictionary
