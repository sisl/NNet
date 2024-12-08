#include <iostream>
#include <cassert>
#include <cmath>
#include <ctime>
#include "nnet.h"

// Helper function to initialize inputs dynamically
double* initializeInputs(int size, double baseValue) {
    double* inputs = new double[size];
    for (int i = 0; i < size; ++i) {
        inputs[i] = baseValue + i * 0.1;  // Incremental values for testing
    }
    return inputs;
}

// Test loading a valid network
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

// Test evaluating the network
void testEvaluateNetwork() {
    const char* valid_filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing evaluation of the network...\n";

    void* network = load_network(valid_filename);
    assert(network != nullptr && "Failed to load a valid network!");

    NNet* nnet = static_cast<NNet*>(network);
    double* input = initializeInputs(nnet->inputSize, 1.0);  // Example base value
    double* output = new double[nnet->outputSize];

    for (int i = 0; i < 5; ++i) {  // Test multiple passes
        for (int j = 0; j < nnet->inputSize; ++j) {
            input[j] += i * 0.1;  // Slightly modify inputs for each test
        }

        int result = evaluate_network(network, input, output, true, true);
        assert(result == 1 && "Network evaluation failed!");

        // Print outputs for debugging
        std::cout << "Pass " << i + 1 << " Outputs: ";
        for (int j = 0; j < nnet->outputSize; ++j) {
            std::cout << output[j] << " ";
        }
        std::cout << "\n";
    }

    delete[] input;
    delete[] output;
    destroy_network(network);
    std::cout << "Network evaluation test passed.\n";
}

// Test loading an invalid network
void testInvalidNetwork() {
    const char* invalid_filename = "../nnet/InvalidNetwork.nnet";
    std::cout << "Testing invalid network loading...\n";

    void* network = load_network(invalid_filename);
    assert(network == nullptr && "Invalid network should not load successfully!");

    std::cout << "Invalid network test passed.\n";
}

// Test timing and memory management
void testTimingAndMemoryManagement() {
    const char* valid_filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing timing and memory management...\n";

    clock_t start = clock();
    void* network = load_network(valid_filename);
    clock_t load_time = clock() - start;

    assert(network != nullptr && "Failed to load a valid network!");

    NNet* nnet = static_cast<NNet*>(network);
    double* input = initializeInputs(nnet->inputSize, 1.0);
    double* output = new double[nnet->outputSize];

    start = clock();
    for (int i = 0; i < 10; ++i) {
        input[0] += 1000.0;  // Modify the first input for variation
        evaluate_network(network, input, output, true, true);
    }
    clock_t eval_time = clock() - start;

    delete[] input;
    delete[] output;
    destroy_network(network);

    std::cout << "Time to load network: " << (load_time * 1000.0 / CLOCKS_PER_SEC) << " ms\n";
    std::cout << "Time to evaluate 10 passes: " << (eval_time * 1000.0 / CLOCKS_PER_SEC) << " ms\n";
    std::cout << "Timing and memory management test passed.\n";
}

// Test edge cases
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
