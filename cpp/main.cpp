#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nnet.h"

void test_network(void)
{
    // Build network and time how long build takes
    const char* filename = "../nnet/TestNetwork.nnet";
    clock_t create_network = clock();
    void *network = load_network(filename);
    clock_t t_build_network = clock() - create_network;
    clock_t start = clock(), diff;

    // Set up variables for evaluating network
    int num_runs = 10;
    double input1[5] = {5299.0,-M_PI,-M_PI,100.0,0.0};
    double output[5] = {0.0,0.0,0.0,0.0,0.0};
    bool normalizeInput = true;
    bool normalizeOutput = true;   
 
    for (int i=0; i<num_runs; i++){
        input1[0] += 1000.0;
        input1[1] += 0.2;
        input1[2] += 0.2;
        input1[3] += 50; 
        input1[4] += 50; 
        evaluate_network(network,input1,output,normalizeInput, normalizeOutput);
        
        // Print out final input/output values
        printf("\nInputs:\n");
        printf("%.3f, %.3f, %.3f, %.3f, %.3f\n",
            input1[0],input1[1],input1[2],input1[3],input1[4]);

        printf("Outputs:\n");
        printf("%.7f, %.7f, %.7f, %.7f %.7f\n",
            output[0],output[1],output[2],output[3],output[4]);
        
    }
    // Stop clock and then destruct network
    diff = clock()-start;
    destroy_network(network); 
    
    // Compute times and print out
    double msec1 = diff*1000.0/CLOCKS_PER_SEC;
    double msec = t_build_network*1000.0/CLOCKS_PER_SEC;
    printf("Time taken to load network:  %.4f milliseconds\n",msec);
    printf("Time taken for %d forward passes: %.4f milliseconds\n",num_runs,msec1);

}

int main(void)
{
    test_network();
    return 0;
}
