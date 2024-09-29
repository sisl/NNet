#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>  // For using the bool type in C
#include "nnet.h"     // Assuming nnet.h provides load_network, evaluate_network, and destroy_network

#define INPUT_SIZE 5  // Define the input and output array sizes

void test_network(void) {
    // Build network and time how long build takes
    const char* filename = "../nnet/TestNetwork.nnet";
    
    // Use clock_gettime for higher resolution timing
    struct timespec start_time, end_time;
    
    // Start timing for loading network
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    void *network = load_network(filename);
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    if (network == NULL) {
        fprintf(stderr, "Error: Failed to load network from file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Calculate time taken to load the network
    double load_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    // Set up variables for evaluating the network
    int num_runs = 10;
    double input1[INPUT_SIZE] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[INPUT_SIZE] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    // Start timing for forward passes
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Run the forward pass multiple times with varying inputs
    for (int i = 0; i < num_runs; i++) {
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50;
        input1[4] += 50;
        
        evaluate_network(network, input1, output, normalizeInput, normalizeOutput);

        // Print out final input/output values
        printf("\nInputs:\n");
        for (int j = 0; j < INPUT_SIZE; j++) {
            printf("%.3f ", input1[j]);
        }
        printf("\nOutputs:\n");
        for (int j = 0; j < INPUT_SIZE; j++) {
            printf("%.7f ", output[j]);
        }
        printf("\n");
    }

    // End timing for forward passes
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate time taken for forward passes
    double forward_pass_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                                  (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    // Destroy the network after use
    destroy_network(network);

    // Print the time measurements
    printf("Time taken to load network:  %.4f milliseconds\n", load_time_ms);
    printf("Time taken for %d forward passes: %.4f milliseconds\n", num_runs, forward_pass_time_ms);
}

int main(void) {
    test_network();
    return 0;
}
