#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>  // For bool type in C
#include "nnet.h"     // Assuming nnet.h provides load_network, evaluate_network, and destroy_network

#define INPUT_SIZE 5  // Define the size of the input array
#define OUTPUT_SIZE 5 // Define the size of the output array

void check_error(int ret_code, const char* msg) {
    if (ret_code != 0) {
        fprintf(stderr, "Error: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

void test_network(void) {
    // Path to the neural network file
    const char* filename = "../nnet/TestNetwork.nnet";
    
    // Time-related variables
    struct timespec start_time, end_time;
    
    // Start timing for loading network
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        perror("Error with clock_gettime (start for loading network)");
        exit(EXIT_FAILURE);
    }
    
    // Load the network
    void *network = load_network(filename);
    
    // End timing for loading network
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        perror("Error with clock_gettime (end for loading network)");
        exit(EXIT_FAILURE);
    }

    // Check if network was loaded successfully
    if (network == NULL) {
        fprintf(stderr, "Error: Failed to load network from file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Calculate time taken to load the network
    double load_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    // Set up variables for evaluating the network
    int num_runs = 10;
    double input[INPUT_SIZE] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[OUTPUT_SIZE] = {0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;

    // Start timing for forward passes
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        perror("Error with clock_gettime (start for forward passes)");
        exit(EXIT_FAILURE);
    }

    // Run the forward pass multiple times with varying inputs
    for (int i = 0; i < num_runs; i++) {
        input[0] += 1000.0;
        input[1] += 0.2;
        input[2] += 0.2;
        input[3] += 50;
        input[4] += 50;

        int eval_success = evaluate_network(network, input, output, normalizeInput, normalizeOutput);
        if (!eval_success) {
            fprintf(stderr, "Error: Network evaluation failed at run %d\n", i + 1);
            destroy_network(network);
            exit(EXIT_FAILURE);
        }

        // Print input and output values
        printf("\nRun %d\nInputs:\n", i + 1);
        for (int j = 0; j < INPUT_SIZE; j++) {
            printf("%.3f ", input[j]);
        }
        printf("\nOutputs:\n");
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%.7f ", output[j]);
        }
        printf("\n");
    }

    // End timing for forward passes
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        perror("Error with clock_gettime (end for forward passes)");
        exit(EXIT_FAILURE);
    }

    // Calculate time taken for forward passes
    double forward_pass_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                                  (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    // Destroy the network after use
    destroy_network(network);

    // Print the time measurements
    printf("\nTime taken to load network:  %.4f milliseconds\n", load_time_ms);
    printf("Time taken for %d forward passes: %.4f milliseconds\n", num_runs, forward_pass_time_ms);
}

int main(void) {
    test_network();
    return 0;
}
