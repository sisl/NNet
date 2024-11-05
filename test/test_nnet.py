import unittest
import numpy as np
from NNet.python.nnet import NNet
import os

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up test environment, ensuring the .nnet file exists."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            self.fail(f"Test file {self.nnetFile} not found!")

    def test_evaluate_valid_input(self):
        """Test evaluating a single valid input vector."""
        nnet = NNet(self.nnetFile)
        input_data = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        output = nnet.evaluate_network(input_data)
        print("Single valid input evaluation output:", output)

    def test_evaluate_multiple_inputs(self):
        """Test evaluating a batch of multiple input vectors."""
        nnet = NNet(self.nnetFile)
        batch_input = np.array([
            [1.0, 1.0, 1.0, 100.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        output = nnet.evaluate_network_multiple(batch_input)
        print("Batch input evaluation output:", output)

    def test_extreme_values(self):
        """Test evaluation with extreme values (inf, nan)."""
        nnet = NNet(self.nnetFile)
        extreme_input = np.array([np.inf, -np.inf, np.nan, 1e10, -1e10], dtype=np.float32)
        
        # Test should check if evaluation handles or raises appropriate errors
        with self.assertRaises(ValueError):
            nnet.evaluate_network(extreme_input)

    def test_large_batch_input(self):
        """Test evaluating a large batch of inputs for network compatibility."""
        nnet = NNet(self.nnetFile)
        large_batch_input = np.random.rand(100, nnet.num_inputs()).astype(np.float32)
        
        # Test network's response to large batch input
        output = nnet.evaluate_network_multiple(large_batch_input)
        print("Large batch input evaluation output shape:", output.shape)
        self.assertEqual(output.shape[0], 100, "Expected 100 outputs for batch input.")

if __name__ == '__main__':
    unittest.main()
