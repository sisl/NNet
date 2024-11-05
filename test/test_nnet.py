import unittest
import numpy as np
from NNet.python.nnet import NNet
import os

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            self.fail(f"Test file {self.nnetFile} not found!")

    def test_evaluate_multiple_inputs(self):
        """Test evaluating multiple inputs in a batch."""
        nnet = NNet(self.nnetFile)
        # Using input size directly without transpose, assuming (N, inputSize)
        batchInput = np.array([
            [1.0, 1.0, 1.0, 100.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        nnetEvalBatch = nnet.evaluate_network_multiple(batchInput)
        print(f"Evaluating multiple inputs: {batchInput}, output: {nnetEvalBatch}")

    def test_extreme_values(self):
        """Test evaluation with extreme values."""
        nnet = NNet(self.nnetFile)
        extremeInput = np.array([np.inf, -np.inf, np.nan, 1e10, -1e10], dtype=np.float32)
        nnetEval = nnet.evaluate_network(extremeInput)
        # Check for finite values in the output or error handling
        self.assertTrue(np.all(np.isfinite(nnetEval)), "Output contains non-finite values.")

    def test_large_batch_input(self):
        """Test evaluating a large batch of inputs."""
        nnet = NNet(self.nnetFile)
        largeBatchInput = np.random.rand(100, nnet.num_inputs()).astype(np.float32)
        nnetEvalBatch = nnet.evaluate_network_multiple(largeBatchInput)
        print(f"Evaluating large batch of inputs, output shape: {nnetEvalBatch.shape}")

if __name__ == '__main__':
    unittest.main()
