#pragma once

//Neural Network Struct
class NNet {
public:
    int numLayers;     //Number of layers in the network
    int inputSize;     //Number of inputs to the network
    int outputSize;    //Number of outputs to the network
    int maxLayerSize;  //Maximum size dimension of a layer in the network
    int *layerSizes;   //Array of the dimensions of the layers in the network

    double *mins;      //Minimum value of inputs
    double *maxes;     //Maximum value of inputs
    double *means;     //Array of the means used to scale the inputs and outputs
    double *ranges;    //Array of the ranges used to scale the inputs and outputs
    double ****matrix; //4D jagged array that stores the weights and biases
                       //the neural network.
    double *inputs;    //Scratch array for inputs to the different layers
    double *temp;      //Scratch array for outputs of different layers
};

//Functions Implemented
extern "C" NNet *load_network(const char *filename);
extern "C" int   num_inputs(void *network);
extern "C" int   num_outputs(void *network);
extern "C" int   evaluate_network(void *network, double *input, double *output, bool normalizeInput, bool normalizeOutput);
extern "C" void  destroy_network(void *network);
