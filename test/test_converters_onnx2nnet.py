import unittest
import os
import numpy as np
from unittest.mock import patch, mock_open
from io import StringIO
from onnx import helper, TensorProto, numpy_helper
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.utils.writeNNet import writeNNet

class TestOnnx2NNet(unittest.TestCase):

    def setUp(self):
        """Set up environment for tests."""
        self.onnxFile = "test_model.onnx"
        self.nnetFile = self.onnxFile.replace(".onnx", ".nnet")

        # Create a simple ONNX model for testing
        input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
        output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        weight = np.random.rand(5, 2).astype(np.float32)
        bias = np.random.rand(2).astype(np.float32)
        weight_initializer = helper.make_tensor("W", TensorProto.FLOAT, weight.shape, weight.flatten())
        bias_initializer = helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten())
        matmul_node = helper.make_node("MatMul", ["X", "W"], ["MatMul_Output"])
        add_node = helper.make_node("Add", ["MatMul_Output", "B"], ["Y"])
        graph = helper.make_graph(
            [matmul_node, add_node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            [weight_initializer, bias_initializer],
        )
        model = helper.make_model(graph)
        helper.save_model(model, self.onnxFile)

    def tearDown(self):
        """Clean up files after tests."""
        for file in [self.onnxFile, self.nnetFile]:
            if os.path.exists(file):
                os.remove(file)

    def test_default_nnet_filename(self):
        """Test default .nnet filename generation."""
        onnx2nnet(self.onnxFile)
        self.assertTrue(os.path.exists(self.nnetFile), "Default .nnet file was not created!")
        os.remove(self.nnetFile)

    def test_provided_nnet_filename(self):
        """Test providing a custom .nnet filename."""
        custom_nnet = "custom_model.nnet"
        onnx2nnet(self.onnxFile, nnetFile=custom_nnet)
        self.assertTrue(os.path.exists(custom_nnet), "Custom .nnet file was not created!")
        os.remove(custom_nnet)

    def test_invalid_node_handling(self):
        """Test handling unsupported ONNX node types."""
        unsupported_node = helper.make_node("UnsupportedOp", ["X"], ["Unsupported_Output"])
        model = helper.load_model(self.onnxFile)
        model.graph.node.append(unsupported_node)
        helper.save_model(model, self.onnxFile)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            onnx2nnet(self.onnxFile)
            output = mock_stdout.getvalue()
            self.assertIn("Unsupported node operation: UnsupportedOp", output)

    def test_writeNNet_function(self):
        """Test the writeNNet function directly."""
        weights = [np.random.rand(5, 2), np.random.rand(2, 1)]
        biases = [np.random.rand(2), np.random.rand(1)]
        inputMins = [-1.0] * 5
        inputMaxes = [1.0] * 5
        means = [0.0] * 6  # 5 inputs + 1 output
        ranges = [1.0] * 6

        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, self.nnetFile)
        self.assertTrue(os.path.exists(self.nnetFile), "Failed to create .nnet file!")

        # Validate .nnet file contents
        with open(self.nnetFile, 'r') as f:
            contents = f.read()
            self.assertIn("Neural Network File Format", contents, "Incorrect .nnet file header!")
            self.assertIn(f"{len(weights)},5,1,", contents, "Incorrect architecture info in .nnet file!")

    def test_mismatched_weights_biases(self):
        """Test error handling for mismatched weights and biases."""
        weights = [np.random.rand(5, 2)]
        biases = [np.random.rand(3)]  # Mismatched length
        inputMins = [-1.0] * 5
        inputMaxes = [1.0] * 5
        means = [0.0] * 6
        ranges = [1.0] * 6

        with self.assertRaises(AssertionError):
            writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, self.nnetFile)

    def test_invalid_onnx_file(self):
        """Test handling of invalid ONNX files."""
        with open("invalid.onnx", "w") as f:
            f.write("not a valid onnx model")

        with self.assertRaises(onnx.OnnxLoadError):
            onnx2nnet("invalid.onnx")

        os.remove("invalid.onnx")

    def test_empty_network(self):
        """Test handling of an empty network."""
        empty_model = helper.make_model(helper.make_graph([], "empty_model", [], []))
        helper.save_model(empty_model, self.onnxFile)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            onnx2nnet(self.onnxFile)
            output = mock_stdout.getvalue()
            self.assertIn("Error: Unable to extract weights and biases", output)


if __name__ == "__main__":
    unittest.main()
