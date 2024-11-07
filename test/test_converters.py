import unittest
import os
import numpy as np
import onnxruntime
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

    def test_missing_file_handling(self):
        """Test handling of a missing input file."""
        missingFile = "nnet/NonExistentFile.nnet"
        with self.assertRaises(FileNotFoundError):
            nnet2onnx(missingFile, "output.onnx")

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

        # Validate ONNX inference
        sess = onnxruntime.InferenceSession(onnxFile, providers=["CPUExecutionProvider"])
        input_shape = sess.get_inputs()[0].shape
        testInput = np.random.rand(*input_shape).astype(np.float32)
        onnxEval = sess.run(None, {sess.get_inputs()[0].name: testInput})[0]

        nnetEval = nnet.evaluate_network(testInput.flatten())
        nnetEval2 = nnet2.evaluate_network(testInput.flatten())

        self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
        np.testing.assert_allclose(onnxEval.flatten(), nnetEval, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
