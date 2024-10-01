import os
import unittest
import numpy as np
import onnx
import onnxruntime
from NNet.converters.nnet2onnx import nnet2onnx
from NNet.converters.onnx2nnet import onnx2nnet
from NNet.converters.pb2nnet import pb2nnet
from NNet.converters.nnet2pb import nnet2pb
from NNet.python.nnet import NNet
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestConverters(unittest.TestCase):

    def test_onnx(self):
        """Test conversion between NNet and ONNX format."""
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32)

        # Convert NNet to ONNX
        onnxFile = nnetFile[:-4] + ".onnx"
        nnet2onnx(nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Convert ONNX back to NNet
        nnetFile2 = nnetFile[:-4] + "v2.nnet"
        onnx2nnet(onnxFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(nnetFile)
        sess = onnxruntime.InferenceSession(onnxFile)
        nnet2 = NNet(nnetFile2)

        # Evaluate ONNX model
        onnxInputName = sess.get_inputs()[0].name
        onnxOutputName = sess.get_outputs()[0].name
        onnxEval = sess.run([onnxOutputName], {onnxInputName: testInput})[0]

        # Evaluate Original NNet model
        inBounds = np.all(testInput >= nnet.mins) and np.all(testInput <= nnet.maxes)
        self.assertTrue(inBounds)
        nnetEval = nnet.evaluate_network(testInput)

        # Evaluate New NNet model
        inBounds = np.all(testInput >= nnet2.mins) and np.all(testInput <= nnet2.maxes)
        self.assertTrue(inBounds)
        nnetEval2 = nnet2.evaluate_network(testInput)

        percChangeONNX = max(abs((nnetEval - onnxEval) / nnetEval)) * 100.0
        percChangeNNet = max(abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Check evaluation consistency
        self.assertTrue(percChangeONNX < 1e-3)
        self.assertTrue(percChangeNNet < 1e-3)

    def test_pb(self):
        """Test conversion between NNet and TensorFlow Protocol Buffer format."""
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32)

        # Convert NNet to TensorFlow PB
        pbFile = nnetFile[:-4] + ".pb"
        nnet2pb(nnetFile, pbFile=pbFile, normalizeNetwork=True)

        # Convert TensorFlow PB back to NNet
        nnetFile2 = nnetFile[:-4] + "v2.nnet"
        pb2nnet(pbFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(nnetFile)
        nnet2 = NNet(nnetFile2)

        # Read the Protocol Buffer file and begin session in TensorFlow 2.x
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")

            placeholders = [x for x in sess.graph.get_operations() if x.node_def.op == 'Placeholder']
            self.assertTrue(len(placeholders) == 1)
            inputName = placeholders[0].name
            outputName = sess.graph.get_operations()[-1].name

            # Evaluate TensorFlow model
            pbEval = sess.run(outputName + ":0", {inputName + ":0": testInput.reshape((1, 5))})[0]

        # Evaluate Original NNet model
        inBounds = np.all(testInput >= nnet.mins) and np.all(testInput <= nnet.maxes)
        self.assertTrue(inBounds)
        nnetEval = nnet.evaluate_network(testInput)

        # Evaluate New NNet model
        inBounds = np.all(testInput >= nnet2.mins) and np.all(testInput <= nnet2.maxes)
        self.assertTrue(inBounds)
        nnetEval2 = nnet2.evaluate_network(testInput)

        percChangePB = max(abs((nnetEval - pbEval) / nnetEval)) * 100.0
        percChangeNNet = max(abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Check evaluation consistency
        self.assertTrue(percChangePB < 1e-3)
        self.assertTrue(percChangeNNet < 1e-3)


if __name__ == '__main__':
    unittest.main()
