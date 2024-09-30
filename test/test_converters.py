import sys
import os
import unittest
import numpy as np
import onnx
import onnxruntime
import tensorflow as tf

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet

# Disable TensorFlow GPU-related logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class TestConverters(unittest.TestCase):

    def setUp(self):
        """Sets up the file paths before each test."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        if not os.path.exists(self.nnetFile):
            self.skipTest(f"Skipping test: {self.nnetFile} does not exist")

    # Add your test cases here...

if __name__ == '__main__':
    unittest.main()
