#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "nnet.h"

// Load the neural network from a .nnet file
NNet* load_network(const char* filename) {
    FILE* fstream = fopen(filename, "r");
    if (fstream == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    int bufferSize = 10240;
    char* buffer = new char[bufferSize];
    if (buffer == NULL) {
        printf("Error: Memory allocation failed for buffer\n");
        fclose(fstream);
        return NULL;
    }

    char* line;
    NNet* nnet = new NNet();
    if (nnet == NULL) {
        printf("Error: Memory allocation failed for NNet\n");
        delete[] buffer;
        fclose(fstream);
        return NULL;
    }

    line = fgets(buffer, bufferSize, fstream);
    while (line != NULL && strstr(line, "//") != NULL) {
        line = fgets(buffer, bufferSize, fstream);
    }

    if (line == NULL) {
        printf("Error: Malformed .nnet file\n");
        delete[] buffer;
        delete nnet;
        fclose(fstream);
        return NULL;
    }

    char* record = strtok(line, ",\n");
    if (record == NULL) {
        printf("Error: Invalid .nnet file format\n");
        delete[] buffer;
        delete nnet;
        fclose(fstream);
        return NULL;
    }

    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL, ",\n"));
    nnet->outputSize = atoi(strtok(NULL, ",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL, ",\n"));

    nnet->layerSizes = new int[nnet->numLayers + 1];
    if (nnet->layerSizes == NULL) {
        printf("Error: Memory allocation failed for layerSizes\n");
        delete[] buffer;
        delete nnet;
        fclose(fstream);
        return NULL;
    }

    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < nnet->numLayers + 1; i++) {
        if (record == NULL) {
            printf("Error: Malformed layer sizes in .nnet file\n");
            delete[] nnet->layerSizes;
            delete[] buffer;
            delete nnet;
            fclose(fstream);
            return NULL;
        }
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL, ",\n");
    }

    delete[] buffer;
    fclose(fstream);
    return nnet;
}

int evaluate_network(void* network, double* input, double* output, bool normalizeInput, bool normalizeOutput) {
    if (network == NULL) {
        printf("Data is Null!\n");
        return -1;
    }

    NNet* nnet = static_cast<NNet*>(network);
    int inputSize = nnet->inputSize;

    if (normalizeInput) {
        for (int i = 0; i < inputSize; i++) {
            if (nnet->ranges[i] == 0) {
                printf("Error: Input range for index %d is zero\n", i);
                return -1;
            }
            if (input[i] > nnet->maxes[i]) {
                nnet->inputs[i] = (nnet->maxes[i] - nnet->means[i]) / nnet->ranges[i];
            } else if (input[i] < nnet->mins[i]) {
                nnet->inputs[i] = (nnet->mins[i] - nnet->means[i]) / nnet->ranges[i];
            } else {
                nnet->inputs[i] = (input[i] - nnet->means[i]) / nnet->ranges[i];
            }
        }
    } else {
        for (int i = 0; i < inputSize; i++) {
            nnet->inputs[i] = input[i];
        }
    }

    return 1;
}

void destroy_network(void* network) {
    NNet* nnet = static_cast<NNet*>(network);
    if (nnet != NULL) {
        for (int i = 0; i < nnet->numLayers; i++) {
            for (int row = 0; row < nnet->layerSizes[i + 1]; row++) {
                delete[] nnet->matrix[i][0][row];
                delete[] nnet->matrix[i][1][row];
            }
            delete[] nnet->matrix[i][0];
            delete[] nnet->matrix[i][1];
            delete[] nnet->matrix[i];
        }
        delete[] nnet->matrix;
        delete[] nnet->layerSizes;
        delete[] nnet->mins;
        delete[] nnet->maxes;
        delete[] nnet->means;
        delete[] nnet->ranges;
        delete[] nnet->inputs;
        delete[] nnet->temp;
        delete nnet;
    }
}
