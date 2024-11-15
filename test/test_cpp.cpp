#include <cassert>
#include <iostream>
#include <cmath>
#include "../nnet.h" 

// Test loading the network
void test_load_network() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(filename);
    assert(network != nullptr && "Failed to load the network");
    std::cout << "test_load_network: PASSED\n";
}

// Test evaluating the network
void test_evaluate_network() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(filename);
    assert(network != nullptr && "Failed to load the network");

    double input[5] = {5299.0, -M_PI, -M_PI, 100.0, 0.0};
    double output[5] = {0.0};

    int result = evaluate_network(network.get(), input, output, true, true);
    assert(result == 1 && "Network evaluation failed");
    
    // Add checks for output values if expected results are known
    std::cout << "Outputs: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";
    std::cout << "test_evaluate_network: PASSED\n";
}

// Test invalid network file handling
void test_load_invalid_network() {
    const char* invalid_filename = "../nnet/InvalidNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(invalid_filename);
    assert(network == nullptr && "Expected load_network to fail with invalid file");
    std::cout << "test_load_invalid_network: PASSED\n";
}

// Test with edge-case inputs
void test_edge_cases() {
    const char* filename = "../nnet/TestNetwork.nnet";
    std::unique_ptr<NNet> network = load_network(filename);
    assert(network != nullptr && "Failed to load the network");

    // Input values at edge cases
    double input[5] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
                       0.0, std::numeric_limits<double>::epsilon(), -std::numeric_limits<double>::max()};
    double output[5] = {0.0};

    int result = evaluate_network(network.get(), input, output, true, true);
    assert(result == 1 && "Network evaluation failed for edge cases");

    std::cout << "test_edge_cases: PASSED\n";
}

// Main test runner
int main() {
    try {
        test_load_network();
        test_evaluate_network();
        test_load_invalid_network();
        test_edge_cases();
        std::cout << "All tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Test failed with unknown exception\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
