import unittest
import os
import numpy as np
import onnxruntime
import tensorflow as tf
from unittest.mock import patch
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet

class TestConverters(unittest.TestCase):

    def setUp(self):
        """Set up the environment by ensuring the required NNet file exists."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def tearDown(self):
        """Clean up generated files after each test."""
        for ext in [".onnx", ".pb", "v2.nnet"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_onnx(self):
        """Test conversion between NNet and ONNX format with edge cases and validation checks."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert NNet to ONNX
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Convert ONNX back to NNet
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load NNet models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        # Load ONNX model for inference
        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])

        # Prepare extreme test inputs to cover edge cases
        testInputs = [
            np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32),
            np.array([-1.0, -1.0, -1.0, -100.0, -1.0], dtype=np.float32),
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
            np.array([1e10, -1e10, 1e-10, -1e-10, 0], dtype=np.float32)
        ]

        for testInput in testInputs:
            # Adjust input shape if required
            input_shape = sess.get_inputs()[0].shape
            if len(input_shape) == 1:
                testInput = testInput.flatten()
            elif len(input_shape) == 2 and input_shape[0] == 1:
                testInput = testInput.reshape(1, -1)

            # Perform inference using ONNX
            onnxInputName = sess.get_inputs()[0].name
            onnxEval = sess.run(None, {onnxInputName: testInput})[0]

            # Evaluate using NNet models
            nnetEval = nnet.evaluate_network(testInput)
            nnetEval2 = nnet2.evaluate_network(testInput)

            # Verify results with increased tolerance
            self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
            np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-3, atol=1e-2)

    def test_pb(self):
        """Test conversion between NNet and TensorFlow Protocol Buffer (PB) format with enhanced checks."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert NNet to PB
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")

        # Convert PB back to NNet
        pb2nnet(pbFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load NNet models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        # Load TensorFlow graph from PB file
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")

            # Retrieve input and output tensors with error handling
            inputTensor = sess.graph.get_tensor_by_name("x:0")
            outputTensor = sess.graph.get_tensor_by_name("y_out:0")

            # Extreme test inputs for PB conversion consistency
            testInputs = [
                np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32),
                np.array([-1.0, -1.0, -1.0, -100.0, -1.0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([1e10, -1e10, 1e-10, -1e-10, 0], dtype=np.float32)
            ]

            for testInput in testInputs:
                testInput = testInput.reshape(1, -1)
                pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]

                # Evaluate using NNet models
                nnetEval = nnet.evaluate_network(testInput.flatten())
                nnetEval2 = nnet2.evaluate_network(testInput.flatten())

                # Verify results with increased tolerance
                self.assertEqual(pbEval.shape, nnetEval.shape, "PB output shape mismatch")
                np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-3, atol=1e-2)

    @patch("sys.exit", side_effect=Exception("FileNotFoundError"))
    def test_invalid_file(self, mock_exit):
        """Test handling of invalid input files."""
        invalid_nnet_file = "invalid_file.nnet"
        with self.assertRaises(Exception) as context:
            nnet2onnx(invalid_nnet_file)
        self.assertEqual(str(context.exception), "FileNotFoundError")

    def test_inconsistent_shapes(self):
        """Test for shape mismatches and input shape handling."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Mismatched shape test
        testInput = np.array([1.0], dtype=np.float32)  # Shape mismatch
        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
        onnxInputName = sess.get_inputs()[0].name
        with self.assertRaises(ValueError):
            sess.run(None, {onnxInputName: testInput})  # Should fail due to shape mismatch

if __name__ == '__main__':
    unittest.main()
