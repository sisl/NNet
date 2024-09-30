import os
import tensorflow as tf
import numpy as np
from NNet.converters.nnet2pb import nnet2pb
from NNet.converters.pb2nnet import pb2nnet
from NNet.python.nnet import NNet
import unittest

class TestConverters(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"  # Ensure this path is correct

    def test_pb(self):
        """Test NNet to TensorFlow Protocol Buffer (PB) conversion and back."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")  # Fixed double dot issue
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert NNet to TensorFlow PB
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)

        # Convert PB back to NNet
        pb2nnet(pbFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        # Load and evaluate TensorFlow model
        graph_def = tf.compat.v1.GraphDef()
        with
