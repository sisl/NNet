#include <iostream>
#include <cassert>
#include <cmath>
#include <ctime>
#include "nnet.h"

void testLoadValidNetwork() {
    const char* valid_filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing valid network loading...\n";

    void* network = load_network(valid_filename);
    assert(network != nullptr && "Failed to load a valid network!");

    NNet* nnet = static_cast<NNet*>(network);

    // Validate network parameters
    assert(nnet->numLayers > 0 && "Number of layers must be greater than zero!");
    assert(nnet->inputSize > 0 && "Input size must be greater than zero!");
    assert(nnet->outputSize > 0 && "Output size must be greater than zero!");
    assert(nnet->layerSizes != nullptr && "Layer sizes array must be allocated!");

    // Validate matrix allocation
    for (int layer = 0; layer < nnet->numLayers; ++layer) {
        for (int row = 0; row < nnet->layerSizes[layer + 1]; ++row) {
            assert(nnet->matrix[layer][0][row] != nullptr && "Weight rows must be allocated!");
            assert(nnet->matrix[layer][1][row] != nullptr && "Bias rows must be allocated!");
        }
    }

    destroy_network(network);
    std::cout << "Valid network loading test passed.\n";
}

void testEvaluateNetwork() {
    const char* valid_filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing evaluation of the network...\n";

    void* network = load_network(valid_filename);
    assert(network != nullptr && "Failed to load a valid network!");

    // Set up inputs, outputs, and normalization flags
    double input1[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    // Evaluate network multiple times with modified inputs
    for (int i = 0; i < 10; ++i) {
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50.0;
        input1[4] += 50.0;

        int result = evaluate_network(network, input1, output, normalizeInput, normalizeOutput);
        assert(result == 1 && "Network evaluation failed!");

        // Print inputs and outputs for debugging
        std::cout << "Inputs: ";
        for (double val : input1) {
            std::cout << val << " ";
        }
        std::cout << "\nOutputs: ";
        for (double val : output) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    destroy_network(network);
    std::cout << "Network evaluation test passed.\n";
}

void testInvalidNetwork() {
    const char* invalid_filename = "../nnet/InvalidNetwork.nnet";
    std::cout << "Testing invalid network loading...\n";

    void* network = load_network(invalid_filename);
    assert(network == nullptr && "Invalid network should not load successfully!");

    std::cout << "Invalid network test passed.\n";
}

void testTimingAndMemoryManagement() {
    const char* valid_filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing timing and memory management...\n";

    clock_t start = clock();
    void* network = load_network(valid_filename);
    clock_t load_time = clock() - start;

    assert(network != nullptr && "Failed to load a valid network!");

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};

    start = clock();
    for (int i = 0; i < 10; ++i) {
        input[0] += 1000.0;
        evaluate_network(network, input, output, true, true);
    }
    clock_t eval_time = clock() - start;

    destroy_network(network);

    std::cout << "Time to load network: " << (load_time * 1000.0 / CLOCKS_PER_SEC) << " ms\n";
    std::cout << "Time to evaluate 10 passes: " << (eval_time * 1000.0 / CLOCKS_PER_SEC) << " ms\n";
    std::cout << "Timing and memory management test passed.\n";
}

void testEdgeCases() {
    std::cout << "Testing edge cases...\n";

    // Test empty file
    const char* empty_filename = "../nnet/EmptyNetwork.nnet";
    void* network = load_network(empty_filename);
    assert(network == nullptr && "Empty network file should not load!");

    // Test malformed network file
    const char* malformed_filename = "../nnet/MalformedNetwork.nnet";
    network = load_network(malformed_filename);
    assert(network == nullptr && "Malformed network file should not load!");

    std::cout << "Edge case tests passed.\n";
}

int main() {
    std::cout << "Running unified test suite for nnet...\n";

    testLoadValidNetwork();
    testEvaluateNetwork();
    testInvalidNetwork();
    testTimingAndMemoryManagement();
    testEdgeCases();

    std::cout << "All tests passed successfully.\n";
    return 0;
}
