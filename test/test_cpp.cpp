#include <iostream>
#include <cassert>
#include <cmath>
#include <ctime>
#include "nnet.h"

void testLoadNetwork() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing loading the network...\n";

    clock_t create_network = clock();
    void* network = load_network(filename);
    clock_t t_build_network = clock() - create_network;

    // Check that the network loaded successfully
    assert(network != nullptr && "Failed to load network!");
    std::cout << "Network loaded successfully.\n";

    // Validate the timing of loading
    double msec_build = t_build_network * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "Time taken to load network: " << msec_build << " milliseconds\n";

    destroy_network(network);
}

void testEvaluateLoop() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::cout << "Testing evaluation loop...\n";

    void* network = load_network(filename);
    assert(network != nullptr && "Failed to load network!");

    // Set up variables for evaluation
    int num_runs = 10;
    double input1[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    clock_t start = clock(), diff;
    for (int i = 0; i < num_runs; i++) {
        // Increment inputs
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50.0;
        input1[4] += 50.0;

        // Print inputs
        std::cout << "\nRunning evaluation " << i + 1 << " with inputs:\n";
        for (double val : input1) {
            std::cout << val << " ";
        }
        std::cout << "\n";

        // Evaluate the network
        int eval_status = evaluate_network(network, input1, output, normalizeInput, normalizeOutput);
        assert(eval_status == 1 && "Network evaluation failed!");

        // Print outputs
        std::cout << "Outputs:\n";
        for (double val : output) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Timing results
    diff = clock() - start;
    double msec_eval = diff * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "Time taken for " << num_runs << " forward passes: " << msec_eval << " milliseconds\n";

    destroy_network(network);
}

int main() {
    std::cout << "Testing nnet package...\n";

    testLoadNetwork();
    testEvaluateLoop();

    std::cout << "All tests passed successfully.\n";
    return 0;
}
