import unittest
import numpy as np
from NNet.python.nnet import NNet
import os

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up the test environment and validate the existence of the .nnet file."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            self.fail(f"Test file {self.nnetFile} not found!")

    def test_evaluate_multiple_inputs(self):
        """Test evaluating multiple inputs in a batch."""
        nnet = NNet(self.nnetFile)
        
        # Construct batch inputs with the correct input size
        batchInput = np.array([
            [1.0, 1.0, 1.0, 100.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Ensure batch input alignment for network processing
        try:
            nnetEvalBatch = nnet.evaluate_network_multiple(batchInput)
            print(f"Batch input evaluation output: {nnetEvalBatch}")
        except ValueError as e:
            self.fail(f"Unexpected ValueError for batch input: {e}")

    def test_large_batch_input(self):
        """Test evaluating a large batch of inputs."""
        nnet = NNet(self.nnetFile)
        
        # Generate a larger batch input with random values
        largeBatchInput = np.random.rand(100, nnet.num_inputs()).astype(np.float32)
        
        # Test the network's ability to process a large batch without errors
        try:
            nnetEvalBatch = nnet.evaluate_network_multiple(largeBatchInput)
            print(f"Large batch input evaluation output shape: {nnetEvalBatch.shape}")
            self.assertEqual(nnetEvalBatch.shape[0], 100, "Expected batch size of 100.")
        except ValueError as e:
            self.fail(f"Unexpected ValueError for large batch input: {e}")

    def test_extreme_values(self):
        """Test evaluation with extreme values."""
        nnet = NNet(self.nnetFile)
        
        # Create an input with extreme values
        extremeInput = np.array([np.inf, -np.inf, np.nan, 1e10, -1e10], dtype=np.float32)
        
        # Attempt evaluation; expect network to handle non-finite values gracefully
        try:
            nnetEval = nnet.evaluate_network(extremeInput)
            self.assertTrue(np.all(np.isfinite(nnetEval)), "Output contains non-finite values.")
        except ValueError:
            print("Network correctly raised ValueError for non-finite input values.")
    
    def test_evaluate_valid(self):
        """Test evaluation with valid input."""
        nnet = NNet(self.nnetFile)
        
        # Test with a simple valid input
        validInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        nnetEval = nnet.evaluate_network(validInput)
        print(f"Evaluation output for valid input: {nnetEval}")
        self.assertEqual(len(nnetEval), nnet.num_outputs(), "Output length mismatch.")

    def test_input_dimension_validation(self):
        """Test network rejects inputs with incorrect dimensions."""
        nnet = NNet(self.nnetFile)
        
        # Attempt to evaluate with an invalid input shape
        invalidInput = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            nnet.evaluate_network(invalidInput)

    def test_empty_file_loading(self):
        """Test behavior when loading an empty .nnet file."""
        emptyFile = "nnet/EmptyNetwork.nnet"
        with open(emptyFile, "w") as f:
            pass  # Create an empty file for testing
        
        with self.assertRaises(ValueError):
            NNet(emptyFile)
        os.remove(emptyFile)

if __name__ == '__main__':
    unittest.main()
