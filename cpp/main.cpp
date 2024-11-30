#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nnet.h"

void test_network(void) {
    // Define the network file path
    const char* filename = "../nnet/TestNetwork.nnet";
    printf("Loading network from file: %s\n", filename);

    // Time the network loading process
    clock_t create_network = clock();
    void* network = load_network(filename);
    clock_t t_build_network = clock() - create_network;

    // Check if the network was loaded successfully
    if (network == NULL) {
        printf("Error: Could not load network from file: %s\n", filename);
        return;
    }
    printf("Network loaded successfully.\n");

    // Set up variables for evaluating the network
    int num_runs = 10;
    double input1[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    // Start timing the forward passes
    clock_t start = clock(), diff;
    for (int i = 0; i < num_runs; i++) {
        // Increment input values for each run
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50.0;
        input1[4] += 50.0;

        // Debug: Print input values
        printf("\nRunning evaluation %d with inputs:\n", i + 1);
        printf("%.3f, %.3f, %.3f, %.3f, %.3f\n",
               input1[0], input1[1], input1[2], input1[3], input1[4]);

        // Evaluate the network
        if (evaluate_network(network, input1, output, normalizeInput, normalizeOutput) == -1) {
            printf("Error: Evaluation failed. Network data might be NULL.\n");
            destroy_network(network);
            return;
        }

        // Debug: Print output values
        printf("Outputs:\n");
        printf("%.7f, %.7f, %.7f, %.7f, %.7f\n",
               output[0], output[1], output[2], output[3], output[4]);
    }

    // Stop timing and destroy the network
    diff = clock() - start;
    destroy_network(network);

    // Compute and print timing results
    double msec_build = t_build_network * 1000.0 / CLOCKS_PER_SEC;
    double msec_eval = diff * 1000.0 / CLOCKS_PER_SEC;
    printf("\nTime taken to load network: %.4f milliseconds\n", msec_build);
    printf("Time taken for %d forward passes: %.4f milliseconds\n", num_runs, msec_eval);
}

int main(void) {
    test_network();
    return 0;
}
