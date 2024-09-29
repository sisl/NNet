import unittest
import sys
import numpy as np
import os
import filecmp
from NNet.python.nnet import NNet
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet
from NNet.utils.normalizeNNet import normalizeNNet

sys.path.append('..')

class TestUtils(unittest.TestCase):

    def test_read(self):
        nnetFile = "nnet/TestNetwork.nnet"
        nnet = NNet(nnetFile)

        # Read the .nnet file using readNNet
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnetFile, withNorm=True)

        # Ensure weights, biases, and other attributes match
        self.assertEqual(len(weights), len(nnet.weights))
        self.assertEqual(len(biases), len(nnet.biases))
        self.assertEqual(len(inputMins), len(nnet.mins))
        self.assertEqual(len(inputMaxes), len(nnet.maxes))
        self.assertEqual(len(means), len(nnet.means))
        self.assertEqual(len(ranges), len(nnet.ranges))

        # Check all elements are equal using numpy's allclose for floating-point comparison
        for w1, w2 in zip(weights, nnet.weights):
            np.testing.assert_allclose(w1, w2)
        for b1, b2 in zip(biases, nnet.biases):
            np.testing.assert_allclose(b1, b2)

        np.testing.assert_allclose(inputMins, nnet.mins)
        np.testing.assert_allclose(inputMaxes, nnet.maxes)
        np.testing.assert_allclose(means, nnet.means)
        np.testing.assert_allclose(ranges, nnet.ranges)

    def test_write(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Load original network
        nnet1 = NNet(nnetFile1)

        # Write it back to a new file
        writeNNet(nnet1.weights, nnet1.biases, nnet1.mins, nnet1.maxes, nnet1.means, nnet1.ranges, nnetFile2)

        # Load the written network and compare
        nnet2 = NNet(nnetFile2)

        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        # Ensure the evaluations match
        np.testing.assert_allclose(eval1, eval2, atol=1e-8)

        # Compare files content-wise
        self.assertTrue(filecmp.cmp(nnetFile1, nnetFile2), "The files should be identical")

        # Cleanup the generated file after the test
        if os.path.exists(nnetFile2):
            os.remove(nnetFile2)

    def test_normalize(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Normalize the network
        normalizeNNet(nnetFile1, nnetFile2)

        # Load both original and normalized network
        nnet1 = NNet(nnetFile1)
        nnet2 = NNet(nnetFile2)

        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        # Check percentage change is within a reasonable threshold
        percChangeNNet = np.max(np.abs((eval1 - eval2) / eval1)) * 100.0
        self.assertLess(percChangeNNet, 1e-3)

        # Cleanup the generated file after the test
        if os.path.exists(nnetFile2):
            os.remove(nnetFile2)

if __name__ == "__main__":
    unittest.main()
