import unittest
import os
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            self.fail(f"Test file {self.nnetFile} not found!")

    def test_evaluate_valid(self):
        """Test evaluation with valid input."""
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        nnet = NNet(self.nnetFile)
        nnetEval = nnet.evaluate_network(testInput)
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

    def test_evaluate_multiple_inputs(self):
        """Test evaluating multiple inputs in a batch."""
        nnet = NNet(self.nnetFile)
        batchInput = np.array([
            [1.0, 1.0, 1.0, 100.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        nnetEvalBatch = nnet.evaluate_network_multiple(batchInput)
        self.assertEqual(nnetEvalBatch.shape, (2, nnet.num_outputs()))

    def test_evaluate_boundary_values(self):
        """Test evaluation with boundary values near zero and large values."""
        nnet = NNet(self.nnetFile)
        boundaryInput = np.array([1e-7, -1e-7, 1e7, -1e7, 1e-7], dtype=np.float32)
        nnetEval = nnet.evaluate_network(boundaryInput)

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
        try:
            with open(emptyFile, "w") as f:
                pass  # Create an empty file
            with self.assertRaises(ValueError):
                NNet(emptyFile)
        finally:
            os.remove(emptyFile)

    def test_invalid_file_format(self):
        """Test loading a file with invalid content."""
        invalidFile = "nnet/InvalidNetwork.nnet"
        try:
            with open(invalidFile, "w") as f:
                f.write("This is not a valid NNet format")
            with self.assertRaises(ValueError):
                NNet(invalidFile)
        finally:
            os.remove(invalidFile)

    def test_large_batch_input(self):
        """Test evaluating a large batch of inputs."""
        nnet = NNet(self.nnetFile)
        largeBatchInput = np.random.rand(100, nnet.num_inputs()).astype(np.float32)
        nnetEvalBatch = nnet.evaluate_network_multiple(largeBatchInput)
        self.assertEqual(nnetEvalBatch.shape, (100, nnet.num_outputs()))

    def test_extreme_values(self):
        """Test evaluation with extreme values."""
        nnet = NNet(self.nnetFile)
        extremeInput = np.array([np.inf, -np.inf, np.nan, 1e10, -1e10], dtype=np.float32)
        with self.assertRaises(ValueError):
            nnet.evaluate_network(extremeInput)

if __name__ == '__main__':
    unittest.main()
