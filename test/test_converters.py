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
import sys

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
            np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-1, atol=2)
            np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-1, atol=2)

    def test_pb(self):
        """Test PB conversion with normalization."""
        self._test_pb_conversion(normalizeNetwork=True)

    def test_pb_without_normalization(self):
        """Test PB conversion without normalization."""
        self._test_pb_conversion(normalizeNetwork=False, compare_direct=False)

    def _test_pb_conversion(self, normalizeNetwork, compare_direct=True):
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=normalizeNetwork)
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")
        pb2nnet(pbFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")
            inputTensor = sess.graph.get_tensor_by_name("x:0")
            outputTensor = sess.graph.get_tensor_by_name("y_out:0")
            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)
            pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        if compare_direct:
            self.assertEqual(pbEval.shape, nnetEval.shape, "PB output shape mismatch")
            np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-2, atol=1e-1)
            np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-2, atol=1e-1)
        else:
            self.assertNotAlmostEqual(np.max(np.abs(nnetEval - pbEval.flatten())), 0, delta=10,
                                      msg="Unexpectedly close values without normalization.")

    def test_invalid_input_file(self):
        """Test handling of an invalid input file in ONNX conversion."""
        with patch("sys.exit", MagicMock()) as mock_exit:
            with self.assertRaises(FileNotFoundError):
                nnet2onnx("invalid_file.nnet", onnxFile="invalid_file.onnx")
            mock_exit.assert_called_once()

    def test_inconsistent_shape(self):
        """Test inference with incorrect input shape for ONNX model."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        testInput = np.array([1.0], dtype=np.float32)
        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
        with self.assertRaises(onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument):
            sess.run(None, {sess.get_inputs()[0].name: testInput})

if __name__ == '__main__':
    unittest.main()
