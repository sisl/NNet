import unittest
import os
import numpy as np
import onnxruntime
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet
import tensorflow as tf

class TestConverters(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def test_onnx(self):
        """Test conversion between NNet and ONNX format."""
        onnxFile = self.nnetFile[:-4] + ".onnx"
        nnetFile2 = self.nnetFile[:-4] + "v2.nnet"

        # Convert NNet to ONNX
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Ensure ONNX file is created
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Convert ONNX back to NNet
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load models and validate
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile)
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # ONNX evaluation
        onnxInputName = sess.get_inputs()[0].name
        onnxEval = sess.run(None, {onnxInputName: testInput.reshape(1, -1)})[0]

        # NNet evaluations
        nnetEval = nnet.evaluate_network(testInput)
        nnetEval2 = nnet2.evaluate_network(testInput)

        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)

    # Similar fix for `test_pb`...
