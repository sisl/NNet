import unittest
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def test_evaluate(self):
        """Test NNet evaluate function."""
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-6, err_msg="Failed on evaluating a sample input.")

        # Test the minimum input point
        testInput = np.array(nnet.mins)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652])
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-6, err_msg="Failed on evaluating minimum input point.")

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-6, err_msg="Failed on evaluating lower than min input point.")

    def test_evaluate_multiple(self):
        """Test NNet evaluate multiple function."""
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.tile(np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32), (3, 1))

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = np.array(nnet.evaluate_network_multiple(testInput))
        outputVal = np.tile(np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737]), (3, 1))
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-6, err_msg="Failed on evaluating multiple inputs.")

        # Test the minimum input point
        testInput = np.tile(np.array(nnet.mins), (3, 1))
        nnetEval = nnet.evaluate_network_multiple(testInput)
        outputVal = np.tile(np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652]), (3, 1))
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-6, err_msg="Failed on evaluating multiple minimum inputs.")

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network_multiple(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-6, err_msg="Failed on evaluating multiple inputs below minimum.")

    def test_num_inputs(self):
        """Test for number of inputs."""
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertTrue(nnet.num_inputs() == 5)

    def test_num_outputs(self):
        """Test for number of outputs."""
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertTrue(nnet.num_outputs() == 5)


if __name__ == '__main__':
    unittest.main()
