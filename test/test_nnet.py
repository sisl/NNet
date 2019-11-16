import unittest
import sys
sys.path.append('..')
import numpy as np
from NNet.python.nnet import NNet

class TestNNet(unittest.TestCase):

    def test_evaluate(self):

        ### Options###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32)
        ##############

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([270.94961805, 280.8974763 , 274.55254776, 288.10071007, 256.18037737])
        self.assertTrue(max(abs(nnetEval-outputVal))<1e-7)

        # Test the minimum input point
        testInput = np.array(nnet.mins)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652])
        self.assertTrue(max(abs(nnetEval-outputVal))<1e-7)

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network(testInput)
        self.assertTrue(max(abs(nnetEval2-outputVal))<1e-7)
        self.assertTrue(max(abs(nnetEval2-nnetEval))<1e-7)

        # Test the maximum input point
        testInput = np.array(nnet.maxes)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647])
        self.assertTrue(max(abs(nnetEval-outputVal))<1e-7)

        # If an input point is greater than the maximum, the maximum is used
        testInput += 1.0
        nnetEval2 = nnet.evaluate_network(testInput)
        self.assertTrue(max(abs(nnetEval2-outputVal))<1e-7)
        self.assertTrue(max(abs(nnetEval2-nnetEval))<1e-7)

    def test_evaluate_multiple(self):

        ### Options###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.tile(np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32),(3,1))
        ##############

        # Load model
        nnet = NNet(nnetFile)
        nnetEval = np.array(nnet.evaluate_network_multiple(testInput))
        outputVal = np.tile(np.array([270.94961805, 280.8974763 , 274.55254776, 288.10071007, 256.18037737]),(3,1))
        self.assertTrue(np.max(abs(nnetEval-outputVal))<1e-7)

        # Test the minimum input point
        testInput = np.tile(np.array(nnet.mins),(3,1))
        nnetEval = nnet.evaluate_network_multiple(testInput)
        outputVal = np.tile(np.array([212.91548563, 211.07829431, 212.55109733, 206.67203848, 211.81989652]),(3,1))
        self.assertTrue(np.max(abs(nnetEval-outputVal))<1e-7)

        # If an input point is less than the minimum, the minimum is used
        testInput -= 1.0
        nnetEval2 = nnet.evaluate_network_multiple(testInput)
        self.assertTrue(np.max(abs(nnetEval2-outputVal))<1e-7)
        self.assertTrue(np.max(abs(nnetEval2-nnetEval))<1e-7)

        # Test the maximum input point
        testInput = np.tile(np.array(nnet.maxes),(3,1))
        nnetEval = nnet.evaluate_network_multiple(testInput)
        outputVal = np.tile(np.array([-0.73022729, 0.43116092, 0.39846494, 0.39557301, 0.38284647]),(3,1))
        self.assertTrue(np.max(abs(nnetEval-outputVal))<1e-7)

        # If an input point is greater than the maximum, the maximum is used
        testInput += 1.0
        nnetEval2 = nnet.evaluate_network_multiple(testInput)
        self.assertTrue(np.max(abs(nnetEval2-outputVal))<1e-7)
        self.assertTrue(np.max(abs(nnetEval2-nnetEval))<1e-7)

    def test_num_inputs(self):
        
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertTrue(nnet.num_inputs()==5)

    def test_num_outputs(self):

        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)
        self.assertTrue(nnet.num_outputs()==5)


if __name__ == '__main__':
    unittest.main()
