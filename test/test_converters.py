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

    def tearDown(self):
        for ext in [".onnx", ".pb", "v2.nnet"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_onnx(self):
        """Test conversion between NNet and ONNX format."""
        onnxFile = self.nnetFile.replace(".nnet", ".onnx")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        try:
            sess = onnxruntime.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
        except Exception as e:
            self.fail(f"Failed to create ONNX inference session: {e}")

        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)

        onnxInputName = sess.get_inputs()[0].name
        try:
            onnxEval = sess.run(None, {onnxInputName: testInput})[0]
        except Exception as e:
            self.fail(f"Failed to run ONNX model inference: {e}")

        nnetEval = nnet.evaluate_network(testInput[0])
        nnetEval2 = nnet2.evaluate_network(testInput[0])

        self.assertEqual(onnxEval.shape, nnetEval.shape, "ONNX output shape mismatch")
        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)

    def test_pb(self):
        """Test conversion between NNet and TensorFlow Protocol Buffer (PB) format."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)
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

            for op in sess.graph.get_operations():
                print(op.name)  # Debug: List all operation names for troubleshooting

            try:
                inputTensor = sess.graph.get_tensor_by_name("input:0")
                outputTensor = sess.graph.get_tensor_by_name("y_out:0")
            except KeyError as e:
                self.fail(f"Tensor not found in graph: {e}")

            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)

            try:
                pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]
            except Exception as e:
                self.fail(f"Failed to run TensorFlow inference: {e}")

        nnetEval = nnet.evaluate_network(testInput[0])
        nnetEval2 = nnet2.evaluate_network(testInput[0])

        self.assertEqual(pbEval.shape, nnetEval.shape, "PB output shape mismatch")
        np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
