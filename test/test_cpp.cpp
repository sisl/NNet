#include <iostream>
#include <cassert>
#include <cmath>
#include "nnet.h"

void test_load_network() {
    const char* testFile = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(testFile);

    assert(network != nullptr);
    std::cout << "Test load_network passed: Network loaded successfully." << std::endl;

    // Ensure network parameters are loaded correctly
    assert(network->numLayers > 0);
    assert(network->inputSize > 0);
    assert(network->outputSize > 0);
    assert(network->layerSizes != nullptr);

    destroy_network(network);
}

void test_num_inputs_outputs() {
    const char* testFile = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(testFile);
    assert(network != nullptr);

    int inputSize = num_inputs(network);
    int outputSize = num_outputs(network);

    assert(inputSize == network->inputSize);
    assert(outputSize == network->outputSize);

    std::cout << "Test num_inputs and num_outputs passed: Inputs = " 
              << inputSize << ", Outputs = " << outputSize << std::endl;

    destroy_network(network);
}

void test_evaluate_network() {
    const char* testFile = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(testFile);
    assert(network != nullptr);

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    int result = evaluate_network(network, input, output, normalizeInput, normalizeOutput);
    assert(result == 1);

    std::cout << "Test evaluate_network passed: Outputs = ";
    for (int i = 0; i < network->outputSize; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    destroy_network(network);
}

void test_destroy_network() {
    const char* testFile = "../nnet/TestNetwork.nnet";
    NNet* network = load_network(testFile);
    assert(network != nullptr);

    destroy_network(network);
    std::cout << "Test destroy_network passed: Memory deallocated successfully." << std::endl;
}

int main() {
    try {
        test_load_network();
        test_num_inputs_outputs();
        test_evaluate_network();
        test_destroy_network();
        std::cout << "All tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "A test failed with exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "A test failed with an unknown error." << std::endl;
    }
    return 0;
}
