import sys
import os
import unittest
import numpy as np
import onnx
import onnxruntime
import tensorflow as tf

# Add the root directory to sys.path to ensure we can import NNet modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Ensure the necessary .nnet file exists
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            self.skipTest(f"Skipping test: {self.nnetFile} does not exist")

    def test_read(self):
        nnet = NNet(self.nnetFile)

        # Read the .nnet file using readNNet
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(self.nnetFile, withNorm=True)

        # Ensure weights, biases, and other attributes match
        self.assertEqual(len(weights), len(nnet.weights))
        self.assertEqual(len(biases), len(nnet.biases))
        self.assertEqual(len(inputMins), len(nnet.mins))
        self.assertEqual(len(inputMaxes), len(nnet.maxes))
        self.assertEqual(len(means), len(nnet.means))
        self.assertEqual(len(ranges), len(nnet.ranges))

        # Check all elements are equal using numpy's allclose for floating-point comparison
        for w1, w2 in zip(weights, nnet.weights):
            np.testing.assert_allclose(w1, w2, rtol=1e-7, atol=1e-8, verbose=True)
        for b1, b2 in zip(biases, nnet.biases):
            np.testing.assert_allclose(b1, b2, rtol=1e-7, atol=1e-8, verbose=True)

        np.testing.assert_allclose(inputMins, nnet.mins, rtol=1e-7, atol=1e-8, verbose=True)
        np.testing.assert_allclose(inputMaxes, nnet.maxes, rtol=1e-7, atol=1e-8, verbose=True)
        np.testing.assert_allclose(means, nnet.means, rtol=1e-7, atol=1e-8, verbose=True)
        np.testing.assert_allclose(ranges, nnet.ranges, rtol=1e-7, atol=1e-8, verbose=True)

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
