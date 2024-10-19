import unittest
import os
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"Test file {self.nnetFile} not found!")

    def test_evaluate(self):
        """Test single input evaluation."""
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        nnet = NNet(self.nnetFile)
        nnetEval = nnet.evaluate_network(testInput)

        print(f"Evaluating input: {testInput}, output: {nnetEval}")

        expectedOutput = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])
        np.testing.assert_allclose(nnetEval, expectedOutput, rtol=1e-5)

    def test_evaluate_min_max(self):
        """Test evaluation with minimum and maximum inputs."""
        nnet = NNet(self.nnetFile)

        # Test with minimum input
        minInput = np.array(nnet.mins)
        minEval = nnet.evaluate_network(minInput)
        print(f"Evaluating minimum input: {minInput}, output: {minEval}")
        expectedMinOutput = np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652])
        np.testing.assert_allclose(minEval, expectedMinOutput, rtol=1e-5)

        # Test with maximum input
        maxInput = np.array(nnet.maxes)
        maxEval = nnet.evaluate_network(maxInput)
        print(f"Evaluating maximum input: {maxInput}, output: {maxEval}")
        expectedMaxOutput = np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647])
        np.testing.assert_allclose(maxEval, expectedMaxOutput, rtol=1e-5)

    def test_evaluate_multiple(self):
        """Test evaluation of multiple inputs."""
        testInput = np.tile(np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32), (3, 1))
        nnet = NNet(self.nnetFile)
        nnetEval = np.array(nnet.evaluate_network_multiple(testInput))

        print(f"Evaluating multiple inputs: {testInput}, output: {nnetEval}")

        expectedOutput = np.tile([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737], (3, 1))
        np.testing.assert_allclose(nnetEval, expectedOutput, rtol=1e-5)

    def test_num_inputs(self):
        """Test the number of inputs."""
        nnet = NNet(self.nnetFile)
        self.assertEqual(nnet.num_inputs(), 5)

    def test_num_outputs(self):
        """Test the number of outputs."""
        nnet = NNet(self.nnetFile)
        self.assertEqual(nnet.num_outputs(), 5)

if __name__ == '__main__':
    unittest.main()
