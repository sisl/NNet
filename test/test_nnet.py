import unittest
import sys
sys.path.append('..')
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def test_evaluate(self):
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        expected_output = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])

        # Load model
        nnet = NNet(nnetFile)
        nnet_eval = nnet.evaluate_network(testInput)
        
        # Assert outputs are almost equal
        np.testing.assert_almost_equal(nnet_eval, expected_output, decimal=7)

        # Test the minimum input point
        testInput = np.array(nnet.mins)
        expected_output_min = np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652])
        nnet_eval_min = nnet.evaluate_network(testInput)
        
        np.testing.assert_almost_equal(nnet_eval_min, expected_output_min, decimal=7)

        # Test if input is below the minimum, output should be same as using min inputs
        testInput_below_min = testInput - 1.0
        nnet_eval_below_min = nnet.evaluate_network(testInput_below_min)
        
        np.testing.assert_almost_equal(nnet_eval_below_min, expected_output_min, decimal=7)

        # Test the maximum input point
        testInput = np.array(nnet.maxes)
        expected_output_max = np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647])
        nnet_eval_max = nnet.evaluate_network(testInput)
        
        np.testing.assert_almost_equal(nnet_eval_max, expected_output_max, decimal=7)

        # Test if input is above the maximum, output should be same as using max inputs
        testInput_above_max = testInput + 1.0
        nnet_eval_above_max = nnet.evaluate_network(testInput_above_max)
        
        np.testing.assert_almost_equal(nnet_eval_above_max, expected_output_max, decimal=7)

    def test_evaluate_multiple(self):
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.tile(np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32), (3, 1))
        expected_output = np.tile(np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737]), (3, 1))

        # Load model
        nnet = NNet(nnetFile)
        nnet_eval = np.array(nnet.evaluate_network_multiple(testInput))

        # Assert outputs are almost equal
        np.testing.assert_almost_equal(nnet_eval, expected_output, decimal=7)

        # Test the minimum input point
        testInput = np.tile(np.array(nnet.mins), (3, 1))
        expected_output_min = np.tile(np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652]), (3, 1))
        nnet_eval_min = nnet.evaluate_network_multiple(testInput)
        
        np.testing.assert_almost_equal(nnet_eval_min, expected_output_min, decimal=7)

        # Test if input is below the minimum, output should be same as using min inputs
        testInput_below_min = testInput - 1.0
        nnet_eval_below_min = nnet.evaluate_network_multiple(testInput_below_min)
        
        np.testing.assert_almost_equal(nnet_eval_below_min, expected_output_min, decimal=7)

        # Test the maximum input point
        testInput = np.tile(np.array(nnet.maxes), (3, 1))
        expected_output_max = np.tile(np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647]), (3, 1))
        nnet_eval_max = nnet.evaluate_network_multiple(testInput)
        
        np.testing.assert_almost_equal(nnet_eval_max, expected_output_max, decimal=7)

        # Test if input is above the maximum, output should be same as using max inputs
        testInput_above_max = testInput + 1.0
        nnet_eval_above_max = nnet.evaluate_network_multiple(testInput_above_max)
        
        np.testing.assert_almost_equal(nnet_eval_above_max, expected_output_max, decimal=7)

    def test_num_inputs(self):
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertEqual(nnet.num_inputs(), 5)

    def test_num_outputs(self):
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertEqual(nnet.num_outputs(), 5)


if __name__ == '__main__':
    unittest.main()
