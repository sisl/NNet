#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include "nnet.h"

// Safely convert strings to double
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
    int layer = 0;

    // Skip comments or header lines
    while (std::getline(file, line) && line.find("//") != std::string::npos) {}

    // Read network parameters
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

    // Read layer sizes
    nnet->layerSizes.resize(nnet->numLayers + 1);
    std::getline(file, line);
    ss.clear();
    ss.str(line);
    for (int i = 0; i <= nnet->numLayers; ++i) {
        std::getline(ss, token, ',');
        nnet->layerSizes[i] = std::stoi(token);
    }

    // Read mins, maxes, means, and ranges for normalization
    auto read_vector = [&](std::vector<double>& vec, int size) {
        vec.resize(size);
        std::getline(file, line);
        ss.clear();
        ss.str(line);
        for (int i = 0; i < size; ++i) {
            std::getline(ss, token, ',');
            vec[i] = safe_atof(token);
        }
    };

    read_vector(nnet->mins, nnet->inputSize);
    read_vector(nnet->maxes, nnet->inputSize);
    read_vector(nnet->means, nnet->inputSize + 1);
    read_vector(nnet->ranges, nnet->inputSize + 1);

    // Initialize network weights and biases
    nnet->matrix.resize(nnet->numLayers);
    for (int l = 0; l < nnet->numLayers; ++l) {
        nnet->matrix[l].resize(2);
        nnet->matrix[l][0].resize(nnet->layerSizes[l + 1], std::vector<double>(nnet->layerSizes[l]));
        nnet->matrix[l][1].resize(nnet->layerSizes[l + 1], std::vector<double>(1));
    }

    // Read weights and biases from file
    for (int l = 0; l < nnet->numLayers; ++l) {
        for (int param = 0; param < 2; ++param) {
            for (int i = 0; i < nnet->layerSizes[l + 1]; ++i) {
                std::getline(file, line);
                ss.clear();
                ss.str(line);
                for (int j = 0; j < nnet->layerSizes[l]; ++j) {
                    std::getline(ss, token, ',');
                    nnet->matrix[l][param][i][j] = safe_atof(token);
                }
            }
        }
    }

    nnet->inputs.resize(nnet->maxLayerSize);
    nnet->temp.resize(nnet->maxLayerSize);

    return nnet;
}

// Deallocate the memory used by the network
void destroy_network(NNet* nnet) {
    // No explicit memory cleanup is needed as we are using smart pointers (unique_ptr).
}

// Evaluate the network with a given input
int evaluate_network(NNet *nnet, const double *input, double *output, bool normalizeInput, bool normalizeOutput) {
    if (!nnet) {
        std::cerr << "Error: Network is NULL!" << std::endl;
        return -1;
    }

    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    // Normalize inputs if required
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

    // Forward pass through each layer
    for (int layer = 0; layer < nnet->numLayers; ++layer) {
        for (int i = 0; i < nnet->layerSizes[layer + 1]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < nnet->layerSizes[layer]; ++j) {
                sum += nnet->inputs[j] * nnet->matrix[layer][0][i][j];
            }
            sum += nnet->matrix[layer][1][i][0];  // Add bias
            nnet->temp[i] = (layer < nnet->numLayers - 1 && sum < 0.0) ? 0.0 : sum;  // ReLU activation
        }
        nnet->inputs = nnet->temp;  // Set inputs for the next layer
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
    const std::string filename = "../nnet/TestNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(filename);

    if (!network) {
        std::cerr << "Failed to load network" << std::endl;
        return -1;
    }

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};

    if (evaluate_network(network.get(), input, output, true, true) == 1) {
        std::cout << "Network evaluated successfully!" << std::endl;
        std::cout << "Output: ";
        for (double val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error evaluating network!" << std::endl;
    }

    return 0;
}
