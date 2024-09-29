#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include "nnet.h"

// Function to safely convert strings to numbers
double safe_atof(const std::string &str) {
    try {
        return std::stod(str);
    } catch (...) {
        return 0.0;
    }
}

// Load the neural network from a .nnet file
std::unique_ptr<NNet> load_network(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filename << std::endl;
        return nullptr;
    }

    auto nnet = std::make_unique<NNet>();
    std::string line;
    int i = 0, layer = 0, row = 0, j = 0, param = 0;

    // Read the network's int parameters
    while (std::getline(file, line) && line.find("//") != std::string::npos) {
        // Skip header lines
    }

    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    nnet->numLayers = std::stoi(tokens[0]);
    nnet->inputSize = std::stoi(tokens[1]);
    nnet->outputSize = std::stoi(tokens[2]);
    nnet->maxLayerSize = std::stoi(tokens[3]);

    // Allocate space and read layer sizes
    nnet->layerSizes.resize(nnet->numLayers + 1);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (i = 0; i < nnet->numLayers + 1; ++i) {
        std::getline(ss, token, ',');
        nnet->layerSizes[i] = std::stoi(token);
    }

    // Read mins, maxes, means, ranges
    nnet->mins.resize(nnet->inputSize);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (i = 0; i < nnet->inputSize; ++i) {
        std::getline(ss, token, ',');
        nnet->mins[i] = safe_atof(token);
    }

    nnet->maxes.resize(nnet->inputSize);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (i = 0; i < nnet->inputSize; ++i) {
        std::getline(ss, token, ',');
        nnet->maxes[i] = safe_atof(token);
    }

    nnet->means.resize(nnet->inputSize + 1);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (i = 0; i < nnet->inputSize + 1; ++i) {
        std::getline(ss, token, ',');
        nnet->means[i] = safe_atof(token);
    }

    nnet->ranges.resize(nnet->inputSize + 1);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (i = 0; i < nnet->inputSize + 1; ++i) {
        std::getline(ss, token, ',');
        nnet->ranges[i] = safe_atof(token);
    }

    // Allocate space for the network matrix
    nnet->matrix.resize(nnet->numLayers);
    for (layer = 0; layer < nnet->numLayers; ++layer) {
        nnet->matrix[layer].resize(2);
        nnet->matrix[layer][0].resize(nnet->layerSizes[layer + 1], std::vector<double>(nnet->layerSizes[layer]));
        nnet->matrix[layer][1].resize(nnet->layerSizes[layer + 1], std::vector<double>(1));
    }

    // Read in the weights and biases
    layer = 0;
    param = 0;
    i = 0;
    j = 0;

    while (std::getline(file, line)) {
        if (i >= nnet->layerSizes[layer + 1]) {
            param = (param == 0) ? 1 : 0;
            if (param == 0) layer++;
            i = 0;
        }

        ss.clear();
        ss.str(line);
        j = 0;
        while (std::getline(ss, token, ',')) {
            nnet->matrix[layer][param][i][j++] = safe_atof(token);
        }
        ++i;
    }

    nnet->inputs.resize(nnet->maxLayerSize);
    nnet->temp.resize(nnet->maxLayerSize);

    return nnet;
}

// Deallocate the memory used by the network
void destroy_network(NNet* nnet) {
    if (!nnet) return;
    // All memory will be automatically freed when unique_ptr goes out of scope.
}

// Evaluate the network with a given input
int evaluate_network(NNet *nnet, double *input, double *output, bool normalizeInput, bool normalizeOutput) {
    if (!nnet) {
        std::cerr << "Error: Network is NULL!" << std::endl;
        return -1;
    }

    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    // Normalize inputs
    for (int i = 0; i < inputSize; ++i) {
        if (normalizeInput) {
            if (input[i] > nnet->maxes[i]) {
                nnet->inputs[i] = (nnet->maxes[i] - nnet->means[i]) / nnet->ranges[i];
            } else if (input[i] < nnet->mins[i]) {
                nnet->inputs[i] = (nnet->mins[i] - nnet->means[i]) / nnet->ranges[i];
            } else {
                nnet->inputs[i] = (input[i] - nnet->means[i]) / nnet->ranges[i];
            }
        } else {
            nnet->inputs[i] = input[i];
        }
    }

    // Forward pass through layers
    for (int layer = 0; layer < nnet->numLayers; ++layer) {
        for (int i = 0; i < nnet->layerSizes[layer + 1]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < nnet->layerSizes[layer]; ++j) {
                sum += nnet->inputs[j] * nnet->matrix[layer][0][i][j];
            }
            sum += nnet->matrix[layer][1][i][0];  // Add bias
            nnet->temp[i] = (layer < nnet->numLayers - 1 && sum < 0.0) ? 0.0 : sum;  // ReLU
        }
        nnet->inputs = nnet->temp;
    }

    // Set final outputs
    for (int i = 0; i < outputSize; ++i) {
        if (normalizeOutput) {
            output[i] = nnet->inputs[i] * nnet->ranges[nnet->inputSize] + nnet->means[nnet->inputSize];
        } else {
            output[i] = nnet->inputs[i];
        }
    }

    return 1;
}

int main() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(filename);

    if (!network) {
        std::cerr << "Failed to load network" << std::endl;
        return -1;
    }

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};

    if (evaluate_network(network.get(), input, output, true, true) == 1) {
        std::cout << "Network evaluated successfully!" << std::endl;
    } else {
        std::cerr << "Error evaluating network!" << std::endl;
    }

    return 0;
}
