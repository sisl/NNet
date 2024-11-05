import unittest
import os
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.nnetFile = "nnet/TestNetwork.nnet"  # Ensure this path points to the correct .nnet file
        self.assertTrue(os.path.exists(self.nnetFile), f"Test file {self.nnetFile} not found!")

    def test_evaluate_valid(self):
        """Test evaluation with valid input."""
        nnet = NNet(self.nnetFile)
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        nnetEval = nnet.evaluate_network(testInput)
        print(f"Evaluating valid input: {testInput}, output: {nnetEval}")

        expectedOutput = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])
        np.testing.assert_allclose(nnetEval, expectedOutput, rtol=1e-5)

    def test_evaluate_invalid_length(self):
        """Test evaluation with incorrect input length."""
        nnet = NNet(self.nnetFile)
        invalidInput = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            nnet.evaluate_network(invalidInput)

    def test_evaluate_out_of_range(self):
        """Test evaluation with out-of-range inputs."""
        nnet = NNet(self.nnetFile)
        outOfRangeInput = np.array([-1000.0, 1000.0, -1000.0, 1000.0, -1000.0], dtype=np.float32)
        nnetEval = nnet.evaluate_network(outOfRangeInput)
        print(f"Evaluating out-of-range input: {outOfRangeInput}, output: {nnetEval}")

    def test_evaluate_multiple_invalid(self):
        """Test multiple input evaluation with invalid shape."""
        nnet = NNet(self.nnetFile)
        invalidBatchInput = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float32).reshape(2, 2)

        with self.assertRaises(ValueError):
            nnet.evaluate_network_multiple(invalidBatchInput)

    def test_num_inputs(self):
        """Test the number of inputs."""
        nnet = NNet(self.nnetFile)
        self.assertEqual(nnet.num_inputs(), 5)

    def test_num_outputs(self):
        """Test the number of outputs."""
        nnet = NNet(self.nnetFile)
        self.assertEqual(nnet.num_outputs(), 5)

    def test_empty_file(self):
        """Test loading an empty NNet file."""
        emptyFile = "nnet/EmptyNetwork.nnet"
        with open(emptyFile, "w") as f:
            pass  # Create an empty file

        with self.assertRaises(ValueError):
            NNet(emptyFile)

        os.remove(emptyFile)

    def test_evaluate_multiple_inputs(self):
        """Test evaluating multiple inputs in a batch."""
        nnet = NNet(self.nnetFile)

        # Create batch input to match the input size of the network
        input_size = nnet.num_inputs()
        batchInput = np.array([
            [1.0] * input_size,  # First input
            [0.0] * input_size   # Second input
        ], dtype=np.float32)

        # Ensure that the input batch shape is valid
        self.assertEqual(batchInput.shape[1], nnet.num_inputs())
        
        try:
            # Run batch evaluation and verify output
            nnetEvalBatch = nnet.evaluate_network_multiple(batchInput)
            print(f"Evaluating multiple inputs: {batchInput}, output: {nnetEvalBatch}")
        except ValueError as e:
            print("Error encountered during batch evaluation:", e)
            self.fail("evaluate_network_multiple raised ValueError unexpectedly.")

if __name__ == '__main__':
    unittest.main()
