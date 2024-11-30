#include <iostream>
#include <cassert>
#include <ctime>
#include <cmath>
#include "nnet.h"

void testLoadAndEvaluateNetwork() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing loading and evaluating the network...\n";

    // Load the network
    clock_t create_network = clock();
    NNet* network = load_network(filename);
    clock_t t_build_network = clock() - create_network;

    assert(network != nullptr && "Failed to load network!");
    std::cout << "Network loaded successfully.\n";

    // Verify network parameters
    assert(network->numLayers > 0 && "Number of layers must be positive!");
    assert(network->inputSize > 0 && "Input size must be positive!");
    assert(network->outputSize > 0 && "Output size must be positive!");

    // Set up variables
    int num_runs = 10;
    double input1[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    // Start timing the forward passes
    clock_t start = clock(), diff;
    for (int i = 0; i < num_runs; i++) {
        // Increment input values
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50.0;
        input1[4] += 50.0;

        // Print input values
        std::cout << "\nRunning evaluation " << (i + 1) << " with inputs:\n";
        for (int j = 0; j < 5; ++j) {
            std::cout << input1[j] << " ";
        }
        std::cout << "\n";

        // Evaluate the network
        int eval_status = evaluate_network(network, input1, output, normalizeInput, normalizeOutput);
        assert(eval_status == 1 && "Evaluation failed!");

        // Print output values
        std::cout << "Outputs:\n";
        for (int j = 0; j < 5; ++j) {
            std::cout << output[j] << " ";
        }
        std::cout << "\n";
    }

    // Stop timing and destroy the network
    diff = clock() - start;
    destroy_network(network);

    // Print timing results
    double msec_build = t_build_network * 1000.0 / CLOCKS_PER_SEC;
    double msec_eval = diff * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "\nTime taken to load network: " << msec_build << " milliseconds\n";
    std::cout << "Time taken for " << num_runs << " forward passes: " << msec_eval << " milliseconds\n";
}

int main() {
    std::cout << "Testing nnet package...\n";

    testLoadAndEvaluateNetwork();

    std::cout << "All tests passed successfully.\n";
    return 0;
}
