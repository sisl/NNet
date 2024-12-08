#pragma once

// Neural Network Struct
class NNet {
public:
    int numLayers;     // Number of layers in the network
    int inputSize;     // Number of inputs to the network
    int outputSize;    // Number of outputs to the network
    int maxLayerSize;  // Maximum size dimension of a layer in the network
    int* layerSizes;   // Array of the dimensions of the layers in the network

    double* mins;      // Minimum value of inputs
    double* maxes;     // Maximum value of inputs
    double* means;     // Array of the means used to scale the inputs and outputs
    double* ranges;    // Array of the ranges used to scale the inputs and outputs
    double**** matrix; // 4D jagged array that stores the weights and biases
                       // of the neural network
    double* inputs;    // Scratch array for inputs to the different layers
    double* temp;      // Scratch array for outputs of different layers
};

// Functions Implemented
extern "C" {
    /**
     * @brief Load a neural network from a .nnet file
     * @param filename Path to the .nnet file
     * @return Pointer to the loaded neural network, or NULL on failure
     */
    NNet* load_network(const char* filename);

    /**
     * @brief Get the number of inputs to the neural network
     * @param network Pointer to the neural network
     * @return Number of inputs, or -1 if the network is NULL
     */
    int num_inputs(void* network);

    /**
     * @brief Get the number of outputs from the neural network
     * @param network Pointer to the neural network
     * @return Number of outputs, or -1 if the network is NULL
     */
    int num_outputs(void* network);

    /**
     * @brief Evaluate the neural network with the given input
     * @param network Pointer to the neural network
     * @param input Array of input values
     * @param output Array to store output values
     * @param normalizeInput Whether to normalize the inputs
     * @param normalizeOutput Whether to normalize the outputs
     * @return 1 on success, -1 on failure
     */
    int evaluate_network(void* network, double* input, double* output,
                         bool normalizeInput, bool normalizeOutput);

    /**
     * @brief Destroy and deallocate memory used by the neural network
     * @param network Pointer to the neural network
     */
    void destroy_network(void* network);
}
