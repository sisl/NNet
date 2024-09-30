import unittest
import sys
import numpy as np
import filecmp
from NNet.python.nnet import NNet
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet
from NNet.utils.normalizeNNet import normalizeNNet

class TestUtils(unittest.TestCase):

    def test_read(self):
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        nnet = NNet(nnetFile)
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnetFile, withNorm=True)

        self.assertEqual(len(weights), len(nnet.weights), "Weights length mismatch")
        self.assertEqual(len(biases), len(nnet.biases), "Biases length mismatch")
        self.assertEqual(len(inputMins), len(nnet.mins), "Input mins length mismatch")
        self.assertEqual(len(inputMaxes), len(nnet.maxes), "Input maxes length mismatch")
        self.assertEqual(len(means), len(nnet.means), "Means length mismatch")
        self.assertEqual(len(ranges), len(nnet.ranges), "Ranges length mismatch")

        for w1, w2 in zip(weights, nnet.weights):
            np.testing.assert_allclose(w1, w2, rtol=1e-7, err_msg="Weight values mismatch")
        for b1, b2 in zip(biases, nnet.biases):
            np.testing.assert_allclose(b1, b2, rtol=1e-7, err_msg="Bias values mismatch")
        np.testing.assert_allclose(inputMins, nnet.mins, rtol=1e-7, err_msg="Input mins mismatch")
        np.testing.assert_allclose(inputMaxes, nnet.maxes, rtol=1e-7, err_msg="Input maxes mismatch")
        np.testing.assert_allclose(means, nnet.means, rtol=1e-7, err_msg="Means mismatch")
        np.testing.assert_allclose(ranges, nnet.ranges, rtol=1e-7, err_msg="Ranges mismatch")

    def test_write(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Load original network and write to new file
        nnet1 = NNet(nnetFile1)
        writeNNet(nnet1.weights, nnet1.biases, nnet1.mins, nnet1.maxes, nnet1.means, nnet1.ranges, nnetFile2)

        # Load the new network
        nnet2 = NNet(nnetFile2)

        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        np.testing.assert_allclose(eval1, eval2, rtol=1e-8, err_msg="Evaluation mismatch after writing")
        
        # File content comparison (may not always work due to floating-point precision issues)
        self.assertTrue(filecmp.cmp(nnetFile1, nnetFile2), "File content mismatch after writing")

    def test_normalize(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Load and normalize network
        nnet1 = NNet(nnetFile1)
        normalizeNNet(nnetFile1, nnetFile2)
        nnet2 = NNet(nnetFile2)

        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        np.testing.assert_allclose(eval1, eval2, rtol=1e-3, err_msg="Evaluation mismatch after normalization")

if __name__ == "__main__":
    unittest.main()
