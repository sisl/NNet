#include <iostream>
#include <cassert>
#include "nnet.h"

void testLoadNetwork() {
    const char* filename = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(filename);
    assert(network != nullptr && "Failed to load network!");
    std::cout << "Network loaded successfully.\n";

    // Check if network parameters are initialized correctly
    assert(network->numLayers > 0 && "Number of layers must be positive!");
    assert(network->inputSize > 0 && "Input size must be positive!");
    assert(network->outputSize > 0 && "Output size must be positive!");
    destroy_network(network);
}

void testEvaluateNetwork() {
    const char* filename = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(filename);
    assert(network != nullptr && "Failed to load network!");

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    int result = evaluate_network(network, input, output, normalizeInput, normalizeOutput);
    assert(result == 1 && "Evaluation failed!");

    std::cout << "Inputs: ";
    for (int i = 0; i < network->inputSize; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << "\nOutputs: ";
    for (int i = 0; i < network->outputSize; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\nEvaluation successful.\n";

    destroy_network(network);
}

void testNumInputsOutputs() {
    const char* filename = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(filename);
    assert(network != nullptr && "Failed to load network!");

    int numInputs = num_inputs(network);
    int numOutputs = num_outputs(network);
    assert(numInputs == network->inputSize && "Input size mismatch!");
    assert(numOutputs == network->outputSize && "Output size mismatch!");

    std::cout << "Number of inputs: " << numInputs << "\n";
    std::cout << "Number of outputs: " << numOutputs << "\n";

    destroy_network(network);
}

int main() {
    std::cout << "Testing nnet package...\n";

    testLoadNetwork();
    testEvaluateNetwork();
    testNumInputsOutputs();

    std::cout << "All tests passed successfully.\n";
    return 0;
}
