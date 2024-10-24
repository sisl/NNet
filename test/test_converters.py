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
        """Set up the environment by ensuring the required NNet file exists."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        if not os.path.exists(self.nnetFile):
            with open(self.nnetFile, "w") as f:
                f.write("Mock network content")  # Simulate a mock NNet file

    def tearDown(self):
        """Clean up generated files after each test."""
        for ext in [".onnx", ".pb", "v2.nnet"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_missing_nnet_file(self):
        """Test handling of a missing NNet file."""
        with self.assertRaises(FileNotFoundError):
            nnet2onnx("missing_file.nnet")

    def test_invalid_onnx_conversion(self):
        """Test ONNX conversion with an invalid NNet file."""
        invalid_nnet_file = "nnet/InvalidNetwork.nnet"
        with open(invalid_nnet_file, "w") as f:
            f.write("Invalid content")

        with self.assertRaises(ValueError):
            nnet2onnx(invalid_nnet_file)

        os.remove(invalid_nnet_file)

    def test_onnx_conversion(self):
        """Test conversion between NNet and ONNX format."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert NNet to ONNX
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Convert ONNX back to NNet
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load models and compare
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        testInput = np.random.rand(1, nnet.num_inputs).astype(np.float32)

        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
        onnxInputName = sess.get_inputs()[0].name
        onnxEval = sess.run(None, {onnxInputName: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)

    def test_pb_conversion(self):
        """Test conversion between NNet and TensorFlow PB format."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert NNet to PB
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")

        # Convert PB back to NNet
        pb2nnet(pbFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load and compare models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        testInput = np.random.rand(1, nnet.num_inputs).astype(np.float32)

        # Load the TensorFlow graph
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")
            inputTensor = sess.graph.get_tensor_by_name("x:0")
            outputTensor = sess.graph.get_tensor_by_name("y_out:0")

            pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)

    def test_unsupported_input_shape(self):
        """Test handling of unsupported input shapes."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile)

        sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
        onnxInputName = sess.get_inputs()[0].name

        # Use an unsupported input shape
        invalidInput = np.random.rand(10, 5).astype(np.float32)
        with self.assertRaises(RuntimeError):
            sess.run(None, {onnxInputName: invalidInput})

if __name__ == '__main__':
    unittest.main()
