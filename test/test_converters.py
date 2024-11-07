import unittest
import os
import numpy as np
import onnx
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
        for ext in [".onnx", ".pb", "v2.nnet", "_custom_output.pb"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_onnx_conversion(self):
        """Test conversion from .nnet to .onnx and back."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        # Convert .nnet to .onnx and back to .nnet
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile, providers=["CPUExecutionProvider"])

        # Prepare the test input based on ONNX input shape
        input_shape = sess.get_inputs()[0].shape
        testInput = np.random.rand(*input_shape).astype(np.float32)
        onnxEval = sess.run(None, {sess.get_inputs()[0].name: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
        np.testing.assert_allclose(
            nnetEval, onnxEval.flatten(), rtol=1e-2, atol=1e-1
        )
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-2, atol=1e-1)

    def test_onnx_invalid_input(self):
        """Test nnet2onnx with invalid input file."""
        invalidFile = "nnet/NonExistentFile.nnet"
        with self.assertRaises(FileNotFoundError):
            nnet2onnx(invalidFile)

    def test_onnx_unsupported_node(self):
        """Test onnx2nnet with unsupported node operation."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Modify ONNX file to introduce an unsupported node
        model = onnx.load(onnxFile)
        unsupported_node = onnx.helper.make_node(
            "UnsupportedOp", inputs=["input"], outputs=["output"]
        )
        model.graph.node.append(unsupported_node)
        onnx.save(model, onnxFile)

        with self.assertRaises(ValueError):
            onnx2nnet(onnxFile)

    def test_onnx_partial_conversion(self):
        """Test partial conversion when weights or biases are missing."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Remove weights or biases from the ONNX file
        model = onnx.load(onnxFile)
        model.graph.initializer.pop(0)  # Remove first initializer
        onnx.save(model, onnxFile)

        with self.assertRaises(ValueError):
            onnx2nnet(onnxFile)

    def test_nnet_to_onnx_no_normalization(self):
        """Test nnet2onnx conversion without normalization."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=False)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

    @patch("onnx.save", side_effect=IOError("Failed to save ONNX model"))
    def test_onnx_save_failure(self, mock_save):
        """Test failure handling when ONNX file cannot be saved."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        with self.assertRaises(IOError):
            nnet2onnx(self.nnetFile, onnxFile=onnxFile)

    @patch("NNet.converters.nnet2onnx.readNNet", side_effect=Exception("Error reading .nnet file"))
    def test_nnet_read_failure(self, mock_read_nnet):
        """Test failure handling when reading .nnet file fails."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        with self.assertRaises(Exception):
            nnet2onnx(self.nnetFile, onnxFile=onnxFile)


if __name__ == "__main__":
    unittest.main()
