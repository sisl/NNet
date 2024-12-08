#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nnet.h"

void test_network(void)
{
    // Path to the network file
    const char* filename = "../nnet/TestNetwork.nnet";

    // Load the network and time the operation
    clock_t create_network = clock();
    void* network = load_network(filename);

    if (network == NULL) {
        printf("Error: Failed to load the network. Exiting test.\n");
        return;
    }

    clock_t t_build_network = clock() - create_network;

    // Cast network to NNet to dynamically handle sizes
    NNet* nnet = static_cast<NNet*>(network);

    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    // Dynamically allocate input and output arrays
    double* input1 = new double[inputSize];
    double* output = new double[outputSize];

    // Initialize inputs with example values
    for (int i = 0; i < inputSize; ++i) {
        input1[i] = (i == 0) ? 5299.0 : (i % 2 == 0 ? -M_PI : 100.0);
    }

    // Perform multiple evaluations
    int num_runs = 10;
    bool normalizeInput = true;
    bool normalizeOutput = true;

    clock_t start = clock();
    for (int i = 0; i < num_runs; ++i) {
        // Modify input values slightly
        input1[0] += 1000.0;
        for (int j = 1; j < inputSize; ++j) {
            input1[j] += (j % 2 == 0) ? 0.2 : 50.0;
        }

        // Evaluate the network
        if (evaluate_network(network, input1, output, normalizeInput, normalizeOutput) != 1) {
            printf("Error: Network evaluation failed in iteration %d.\n", i + 1);
            continue;
        }

        // Print inputs and outputs
        printf("\nInputs:\n");
        for (int j = 0; j < inputSize; ++j) {
            printf("%.3f ", input1[j]);
        }
        printf("\nOutputs:\n");
        for (int j = 0; j < outputSize; ++j) {
            printf("%.7f ", output[j]);
        }
        printf("\n");
    }
    clock_t diff = clock() - start;

    // Cleanup
    destroy_network(network);
    delete[] input1;
    delete[] output;

    // Print timing results
    double msec1 = diff * 1000.0 / CLOCKS_PER_SEC;
    double msec = t_build_network * 1000.0 / CLOCKS_PER_SEC;
    printf("Time taken to load network:  %.4f milliseconds\n", msec);
    printf("Time taken for %d forward passes: %.4f milliseconds\n", num_runs, msec1);
}

int main(void)
{
    test_network();
    return 0;
}
