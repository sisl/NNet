import unittest
import os
import numpy as np
import onnxruntime
from unittest.mock import patch
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.python.nnet import NNet


class TestConverters(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def tearDown(self):
        for ext in [".onnx", "v2.nnet"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_onnx(self):
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])

        input_shape = sess.get_inputs()[0].shape
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)
        onnxEval = sess.run(None, {sess.get_inputs()[0].name: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-3, atol=1e-2)

    @patch("NNet.converters.nnet2onnx.readNNet", side_effect=FileNotFoundError)
    def test_missing_nnet_file(self, mock_read):
        with self.assertRaises(FileNotFoundError):
            nnet2onnx("missing_file.nnet")
