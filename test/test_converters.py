import unittest
import os
import numpy as np
import onnxruntime
from unittest.mock import patch, MagicMock
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

    def tearDown(self):
        for ext in [".onnx", ".pb", "v2.nnet", "_custom_output.pb"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_onnx(self):
        """Test conversion between NNet and ONNX format with edge cases."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load models and perform inference with validation
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])

        testInputs = [
            np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-1.0, -1.0, -1.0, -100.0, -1.0], dtype=np.float32)
        ]

        for testInput in testInputs:
            input_shape = sess.get_inputs()[0].shape
            if len(input_shape) == 2 and input_shape[0] == 1:
                testInput = testInput.reshape(1, -1)

            onnxEval = sess.run(None, {sess.get_inputs()[0].name: testInput})[0]
            nnetEval = nnet.evaluate_network(testInput.flatten())
            nnetEval2 = nnet2.evaluate_network(testInput.flatten())

            self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
            np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=0.3, atol=10)
            np.testing.assert_allclose(nnetEval, nnetEval2, rtol=0.3, atol=10)

    def test_invalid_input_file(self):
        """Test handling of an invalid input file in ONNX conversion."""
        with self.assertRaises(FileNotFoundError):
            nnet2onnx("invalid_file.nnet", onnxFile="invalid_file.onnx")

    # Other tests remain unchanged

if __name__ == '__main__':
    unittest.main()
