import unittest
import sys
sys.path.append('..')
import numpy as np
import filecmp
from NNet.python.nnet import *
from NNet.utils.readNNet import readNNet
from NNet.utils.writeNNet import writeNNet
from NNet.utils.normalizeNNet import normalizeNNet

class TestUtils(unittest.TestCase):

    def test_read(self):

        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32)
        nnet = NNet(nnetFile)
        weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnetFile, withNorm=True)

        self.assertTrue(len(weights)==len(nnet.weights))
        self.assertTrue(len(biases)==len(nnet.biases))
        self.assertTrue(len(inputMins)==len(nnet.mins))
        self.assertTrue(len(inputMaxes)==len(nnet.maxes))
        self.assertTrue(len(means)==len(nnet.means))
        self.assertTrue(len(ranges)==len(nnet.ranges))
        for w1, w2 in zip(weights,nnet.weights):
        	self.assertTrue(np.all(w1==w2))
        for b1, b2 in zip(biases,nnet.biases):
        	self.assertTrue(np.all(b1==b2))
        self.assertTrue(np.all(inputMins==nnet.mins))
        self.assertTrue(np.all(inputMaxes==nnet.maxes))
        self.assertTrue(np.all(means==nnet.means))
        self.assertTrue(np.all(ranges==nnet.ranges))

    def test_write(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32)
        nnet1 = NNet(nnetFile1)
        writeNNet(nnet1.weights,nnet1.biases,nnet1.mins,nnet1.maxes,nnet1.means,nnet1.ranges,nnetFile2)
        nnet2 = NNet(nnetFile2)

        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        percChangeNNet = max(abs((eval1-eval2)/eval1))*100.0
        self.assertTrue(percChangeNNet<1e-8)
        self.assertTrue(filecmp.cmp(nnetFile1,nnetFile2))

    def test_normalize(self):
        nnetFile1 = "nnet/TestNetwork.nnet"
        nnetFile2 = "nnet/TestNetwork.v2.nnet"
        testInput = np.array([1.0,1.0,1.0,100.0,1.0]).astype(np.float32)
        nnet1 = NNet(nnetFile1)
        normalizeNNet(nnetFile1,nnetFile2)
        nnet2 = NNet(nnetFile2)
        eval1 = nnet1.evaluate_network(testInput)
        eval2 = nnet2.evaluate_network(testInput)

        percChangeNNet = max(abs((eval1-eval2)/eval1))*100.0
        self.assertTrue(percChangeNNet<1e-3)


if __name__ == "__main__":
	unittest.main()