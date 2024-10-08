import unittest
import os
import numpy as np
import onnxruntime
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet
from NNet.utils.writeNNet import writeNNet  # Import the writeNNet function
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class TestConverters(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def test_onnx(self):
        """Test conversion between NNet and ONNX format."""
        onnxFile = self.nnetFile[:-4] + ".onnx"
        nnetFile2 = self.nnetFile[:-4] + "v2.nnet"

        # Convert NNet to ONNX
        nnet2onnx(self.nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Ensure ONNX file is created
        self.assertTrue(os.path.exists(onnxFile), f"{onnxFile} not found!")

        # Convert ONNX back to NNet
        onnx2nnet(onnxFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load models and validate
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        sess = onnxruntime.InferenceSession(onnxFile)
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)

        # ONNX evaluation
        onnxInputName = sess.get_inputs()[0].name
        onnxEval = sess.run(None, {onnxInputName: testInput})[0]

        # NNet evaluations
        nnetEval = nnet.evaluate_network(testInput[0])  # NNet expects 1D array
        nnetEval2 = nnet2.evaluate_network(testInput[0])

        np.testing.assert_allclose(nnetEval, onnxEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)

    def test_pb(self):
        """Test conversion between NNet and TensorFlow Protocol Buffer (PB) format."""
        pbFile = self.nnetFile[:-4] + ".pb"
        nnetFile2 = self.nnetFile[:-4] + "v2.nnet"

        # Convert NNet to TensorFlow PB
        nnet2pb(self.nnetFile, pbFile=pbFile, normalizeNetwork=True)

        # Ensure PB file is created
        self.assertTrue(os.path.exists(pbFile), f"{pbFile} not found!")

        # Convert TensorFlow PB back to NNet
        pb2nnet(pbFile, nnetFile=nnetFile2)
        self.assertTrue(os.path.exists(nnetFile2), f"{nnetFile2} not found!")

        # Load models and validate
        nnet = NNet(self.nnetFile)
        nnet2 = NNet(nnetFile2)
        
        # Read TensorFlow PB file and evaluate the model using TensorFlow
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")
            
            inputTensor = sess.graph.get_tensor_by_name("input:0")  # Adjust name if needed
            outputTensor = sess.graph.get_tensor_by_name("y_out:0")  # Adjust name if needed

            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)
            pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]

        # NNet evaluations
        nnetEval = nnet.evaluate_network(testInput[0])  # NNet expects 1D array
        nnetEval2 = nnet2.evaluate_network(testInput[0])

        np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-5)
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)


class TestNNet(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def test_evaluate(self):
        """Test the evaluation of a single input"""
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

        # Load model
        nnet = NNet(self.nnetFile)
        nnetEval = nnet.evaluate_network(testInput)
        outputVal = np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737])

        # Check output within tolerance
        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-5)

    def test_evaluate_multiple(self):
        """Test the evaluation of multiple inputs"""
        testInput = np.tile(np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32), (3, 1))

        # Load model
        nnet = NNet(self.nnetFile)
        nnetEval = np.array(nnet.evaluate_network_multiple(testInput))
        outputVal = np.tile(np.array([270.94961805, 280.8974763, 274.55254776, 288.10071007, 256.18037737]), (3, 1))

        np.testing.assert_allclose(nnetEval, outputVal, rtol=1e-5)


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.nnetFile1 = "nnet/TestNetwork.nnet"
        self.nnetFile2 = "nnet/TestNetwork.v2.nnet"
        self.assertTrue(os.path.exists(self.nnetFile1), f"{self.nnetFile1} not found!")
        self.testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)

    def test_write(self):
        """Test writing a NNet model to a file and comparing outputs."""
        nnet1 = NNet(self.nnetFile1)
        writeNNet(nnet1.weights, nnet1.biases, nnet1.mins, nnet1.maxes, nnet1.means, nnet1.ranges, self.nnetFile2)
        nnet2 = NNet(self.nnetFile2)

        eval1 = nnet1.evaluate_network(self.testInput)
        eval2 = nnet2.evaluate_network(self.testInput)

        np.testing.assert_allclose(eval1, eval2, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
