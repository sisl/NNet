import unittest
import numpy as np
import onnx
import onnxruntime
import tensorflow as tf
import os

from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet

# Disable TensorFlow GPU-related logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class TestConverters(unittest.TestCase):

    def setUp(self):
        """Sets up the file paths before each test."""
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        if not os.path.exists(self.nnetFile):
            self.skipTest(f"Skipping test: {self.nnetFile} does not exist")

    def test_onnx(self):
        """Test NNet to ONNX conversion and back."""
        onnxFile = self.nnetFile[:-4] + ".onnx"
        nnetFile2 = self.nnetFile[:-4] + "v2.nnet"

        # Convert NNet to ONNX
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Convert ONNX back to NNet
        onnx2nnet(onnxFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile)

        # Evaluate ONNX
        onnxInputName = sess.get_inputs()[0].name
        onnxOutputName = sess.get_outputs()[0].name
        onnxEval = sess.run([onnxOutputName], {onnxInputName: self.testInput})[0]

        # Evaluate Original and Converted NNet
        nnetEval = nnet.evaluate_network(self.testInput)
        nnetEval2 = nnet2.evaluate_network(self.testInput)

        percChangeONNX = np.max(np.abs((nnetEval - onnxEval) / nnetEval)) * 100.0
        percChangeNNet = np.max(np.abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Assert evaluation consistency
        self.assertLess(percChangeONNX, 1e-3)
        self.assertLess(percChangeNNet, 1e-3)

    def test_pb(self):
        """Test NNet to TensorFlow Protocol Buffer (PB) conversion and back."""
        pbFile = self.nnetFile[:-4] + ".pb"
        nnetFile2 = self.nnetFile[:-4] + "v2.nnet"

        # Convert NNet to TensorFlow PB
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)

        # Convert PB back to NNet
        pb2nnet(pbFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)

        # Load and evaluate TensorFlow model
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        input_tensor = graph.get_tensor_by_name("input:0")
        output_tensor = graph.get_tensor_by_name("y_out:0")

        # Evaluate using TensorFlow
        with tf.compat.v1.Session(graph=graph) as sess:
            pbEval = sess.run(output_tensor, feed_dict={input_tensor: self.testInput.reshape(1, -1)})[0]

        # Evaluate Original and Converted NNet
        nnetEval = nnet.evaluate_network(self.testInput)
        nnetEval2 = nnet2.evaluate_network(self.testInput)

        percChangePB = np.max(np.abs((nnetEval - pbEval) / nnetEval)) * 100.0
        percChangeNNet = np.max(np.abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Assert evaluation consistency
        self.assertLess(percChangePB, 1e-3)
        self.assertLess(percChangeNNet, 1e-3)


if __name__ == '__main__':
    unittest.main()
