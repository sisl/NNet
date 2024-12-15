import unittest
import os
import numpy as np
from NNet.python.nnet import NNet
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet
from NNet.utils.normalizeNNet import normalizeNNet

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.nnetFile1 = "nnet/TestNetwork.nnet"
        self.nnetFile2 = "nnet/TestNetwork.v2.nnet"
        self.assertTrue(os.path.exists(self.nnetFile1), f"Test file {self.nnetFile1} not found!")
        self.testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

    def test_read(self):
        """Test reading a NNet file and comparing with NNet object."""
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(self.nnetFile1, withNorm=True)
        nnet = NNet(self.nnetFile1)

        self.assertEqual(len(weights), len(nnet.weights))
        self.assertEqual(len(biases), len(nnet.biases))
        self.assertEqual(len(inputMins), len(nnet.mins))
        self.assertEqual(len(inputMaxes), len(nnet.maxes))
        self.assertEqual(len(means), len(nnet.means))
        self.assertEqual(len(ranges), len(nnet.ranges))

        for w1, w2 in zip(weights, nnet.weights):
            self.assertTrue(np.allclose(w1, w2, rtol=1e-5))
        for b1, b2 in zip(biases, nnet.biases):
            self.assertTrue(np.allclose(b1, b2, rtol=1e-5))
        self.assertTrue(np.allclose(inputMins, nnet.mins, rtol=1e-5))
        self.assertTrue(np.allclose(inputMaxes, nnet.maxes, rtol=1e-5))
        self.assertTrue(np.allclose(means, nnet.means, rtol=1e-5))
        self.assertTrue(np.allclose(ranges, nnet.ranges, rtol=1e-5))

    def test_write(self):
        """Test writing a NNet model to a file and comparing outputs."""
        nnet1 = NNet(self.nnetFile1)
        writeNNet(nnet1.weights, nnet1.biases, nnet1.mins, nnet1.maxes, nnet1.means, nnet1.ranges, self.nnetFile2)
        nnet2 = NNet(self.nnetFile2)

        eval1 = nnet1.evaluate_network(self.testInput)
        eval2 = nnet2.evaluate_network(self.testInput)

        np.testing.assert_allclose(eval1, eval2, rtol=1e-5)

    def test_normalize(self):
        """Test normalization of a NNet model."""
        nnet1 = NNet(self.nnetFile1)
        normalizeNNet(self.nnetFile1, self.nnetFile2)
        nnet2 = NNet(self.nnetFile2)

        eval1 = nnet1.evaluate_network(self.testInput)
        eval2 = nnet2.evaluate_network(self.testInput)

        np.testing.assert_allclose(eval1, eval2, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
