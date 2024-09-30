import unittest
import sys
sys.path.append('..')
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def test_evaluate(self):
        ### Options ###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32)
        ###############

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating a sample input.")

        # Test the minimum input point
        testInput = np.array(nnet.mins)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652])
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating minimum input point.")

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-7, err_msg="Failed on evaluating lower than min input point.")

        # Test the maximum input point
        testInput = np.array(nnet.maxes)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647])
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating max input point.")

        # If an input point is greater than the maximum, the maximum is used
        testInput += 1.0
        nnetEval2 = nnet.evaluate_network(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-7, err_msg="Failed on evaluating greater than max input point.")

    def test_evaluate_multiple(self):
        ### Options ###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.tile(np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32), (3, 1))
        ###############

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = np.array(nnet.evaluate_network_multiple(testInput))
        outputVal = np.tile(np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737]), (3, 1))
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating multiple inputs.")

        # Test the minimum input point
        testInput = np.tile(np.array(nnet.mins), (3, 1))
        nnetEval = nnet.evaluate_network_multiple(testInput)
        outputVal = np.tile(np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652]), (3, 1))
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating multiple minimum inputs.")

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network_multiple(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-7, err_msg="Failed on evaluating multiple inputs below minimum.")

        # Test the maximum input point
        testInput = np.tile(np.array(nnet.maxes), (3, 1))
        nnetEval = nnet.evaluate_network_multiple(testInput)
        outputVal = np.tile(np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647]), (3, 1))
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-7, err_msg="Failed on evaluating multiple max inputs.")

        # If an input point is greater than the maximum, the maximum is used
        testInput += 1.0
        nnetEval2 = nnet.evaluate_network_multiple(testInput)
        np.testing.assert_allclose(nnetEval2, outputVal, rtol=1e-7, err_msg="Failed on evaluating multiple inputs above maximum.")

    def test_num_inputs(self):
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertEqual(nnet.num_inputs(), 5, "Number of inputs mismatch.")

    def test_num_outputs(self):
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertEqual(nnet.num_outputs(), 5, "Number of outputs mismatch.")


if __name__ == '__main__':
    unittest.main()
