import unittest
import os
import numpy as np
import onnxruntime
from unittest.mock import patch
from io import StringIO
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

    # ONNX Tests
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

        # Prepare the test input based on ONNX input shape
        input_shape = sess.get_inputs()[0].shape
        if len(input_shape) == 1:
            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        elif len(input_shape) == 2:
            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)

        onnxEval = sess.run(None, {sess.get_inputs()[0].name: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-3, atol=1e-2)

    def test_default_onnx_filename(self):
        nnet2onnx(self.nnetFile)  # No onnxFile specified
        default_onnx_file = self.nnetFile.replace(".nnet", ".onnx")
        self.assertTrue(os.path.exists(default_onnx_file), f"Default ONNX file {default_onnx_file} not created!")

    def test_file_not_found(self):
        non_existent_file = "non_existent.nnet"
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as excinfo:
                nnet2onnx(non_existent_file)
            self.assertEqual(excinfo.exception.code, 1)  # Verify SystemExit with code 1

        output = mock_stdout.getvalue()
        self.assertIn("Error: The file non_existent.nnet was not found.", output)

    @patch("sys.argv", ["nnet2onnx.py", "nnet/TestNetwork.nnet", "--normalize"])
    def test_argparse_execution(self):
        from NNet.converters.nnet2onnx import main
        main()
        default_onnx_file = self.nnetFile.replace(".nnet", ".onnx")
        self.assertTrue(os.path.exists(default_onnx_file), "Default ONNX file not created via argparse!")

    # PB Tests
    def test_normalized_pb_conversion(self):
        """Test PB conversion with normalized weights and biases."""
        pbFile = self.nnetFile.replace(".nnet", "_normalized.pb")
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")
        os.remove(pbFile)  # Cleanup

    def test_pb_with_custom_output_node(self):
        """Test PB conversion with a custom output node name."""
        pbFile = self.nnetFile.replace(".nnet", "_custom_output.pb")
        nnet2pb(self.nnetFile, pbFile=pbFile, output_node_names="custom_output")
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not created!")

        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        nodes = [node.name for node in graph_def.node]
        self.assertIn("custom_output", nodes, "Custom output node not found in the frozen graph!")
        os.remove(pbFile)  # Cleanup

    @patch("NNet.utils.readNNet.readNNet", side_effect=FileNotFoundError("File not found"))
    def test_invalid_file_read(self, mock_readNNet):
        """Test behavior when reading an invalid .nnet file."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with self.assertRaises(FileNotFoundError):
                nnet2pb(self.nnetFile, pbFile=pbFile)
            output = mock_stdout.getvalue()
            self.assertIn("File not found", output)

    def test_model_layer_building(self):
        """Test the model layer-by-layer building."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        nnet2pb(self.nnetFile, pbFile=pbFile)
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")

        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Check if the graph contains expected nodes
        layers = [node.name for node in graph_def.node]
        self.assertIn("y_out", layers, "Output node y_out not found in the frozen graph!")
        os.remove(pbFile)  # Cleanup


if __name__ == "__main__":
    unittest.main()
