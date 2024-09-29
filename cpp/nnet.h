#pragma once

#include <vector>
#include <memory>

// Neural Network Struct
class NNet {
public:
    int numLayers;     // Number of layers in the network
    int inputSize;     // Number of inputs to the network
    int outputSize;    // Number of outputs to the network
    int maxLayerSize;  // Maximum size dimension of a layer in the network

    std::vector<int> layerSizes;   // Vector of the dimensions of the layers in the network

    std::vector<double> mins;      // Minimum value of inputs
    std::vector<double> maxes;     // Maximum value of inputs
    std::vector<double> means;     // Vector of the means used to scale the inputs and outputs
    std::vector<double> ranges;    // Vector of the ranges used to scale the inputs and outputs

    // 4D jagged array that stores the weights and biases for the neural network
    std::vector<std::vector<std::vector<std::vector<double>>>> matrix;

    std::vector<double> inputs;    // Scratch vector for inputs to the different layers
    std::vector<double> temp;      // Scratch vector for outputs of different layers

    NNet() = default;              // Default constructor
    ~NNet() = default;             // Default destructor
};

// Functions implemented in C, exposed via C linkage
extern "C" std::unique_ptr<NNet> load_network(const char *filename);
extern "C" int num_inputs(const NNet* network);
extern "C" int num_outputs(const NNet* network);
extern "C" int evaluate_network(NNet* network, double *input, double *output, bool normalizeInput, bool normalizeOutput);
extern "C" void destroy_network(NNet* network);

