import unittest
import sys
import numpy as np
import onnx
import os
import tensorflow as tf
import onnxruntime

# Adjust import paths based on your folder structure
from converters.nnet2onnx import nnet2onnx
from converters.onnx2nnet import onnx2nnet
from converters.pb2nnet import pb2nnet
from converters.nnet2pb import nnet2pb
from python.nnet import NNet

# Disable TensorFlow GPU-related logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class TestConverters(unittest.TestCase):

    def test_onnx(self):
        ### Options ###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32)

        # Convert NNET to ONNX
        onnxFile = nnetFile[:-4] + "onnx"
        nnet2onnx(nnetFile, onnxFile=onnxFile, normalizeNetwork=True)

        # Convert ONNX back to NNET
        nnetFile2 = nnetFile[:-4] + "v2.nnet"
        onnx2nnet(onnxFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(nnetFile)
        sess = onnxruntime.InferenceSession(onnxFile)
        nnet2 = NNet(nnetFile2)

        # Evaluate ONNX
        onnxInputName = sess.get_inputs()[0].name
        onnxOutputName = sess.get_outputs()[0].name
        onnxEval = sess.run([onnxOutputName], {onnxInputName: testInput})[0]

        # Evaluate Original NNET
        self.assertTrue(np.all(testInput >= nnet.mins) and np.all(testInput <= nnet.maxes))
        nnetEval = nnet.evaluate_network(testInput)

        # Evaluate New NNET
        self.assertTrue(np.all(testInput >= nnet2.mins) and np.all(testInput <= nnet2.maxes))
        nnetEval2 = nnet2.evaluate_network(testInput)

        percChangeONNX = np.max(np.abs((nnetEval - onnxEval) / nnetEval)) * 100.0
        percChangeNNet = np.max(np.abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Evaluation should not change
        self.assertTrue(percChangeONNX < 1e-3)
        self.assertTrue(percChangeNNet < 1e-3)

    def test_pb(self):
        ### Options ###
        nnetFile = "nnet/TestNetwork.nnet"
        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0]).astype(np.float32)

        pbFile = nnetFile[:-4] + "pb"

        # Check if the .pb file exists, skip the test if not found
        if not os.path.exists(pbFile):
            self.skipTest(f"Skipping test because {pbFile} is not found")

        # Convert NNET to TensorFlow PB
        nnet2pb(nnetFile, pbFile=pbFile, normalizeNetwork=True)

        # Convert PB back to NNET
        nnetFile2 = nnetFile[:-4] + "v2.nnet"
        pb2nnet(pbFile, nnetFile=nnetFile2)

        # Load models
        nnet = NNet(nnetFile)
        nnet2 = NNet(nnetFile2)

        # Load and evaluate TensorFlow model
        tf.compat.v1.reset_default_graph()  # Needed if running multiple times in a session
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pbFile, "rb") as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        input_tensor = graph.get_tensor_by_name("input:0")
        output_tensor = graph.get_tensor_by_name("y_out:0")

        # Evaluate using TensorFlow
        with tf.compat.v1.Session(graph=graph) as sess:
            tf_eval = sess.run(output_tensor, feed_dict={input_tensor: testInput.reshape(1, -1)})[0]

        # Evaluate Original NNET
        self.assertTrue(np.all(testInput >= nnet.mins) and np.all(testInput <= nnet.maxes))
        nnetEval = nnet.evaluate_network(testInput)

        # Evaluate New NNET
        self.assertTrue(np.all(testInput >= nnet2.mins) and np.all(testInput <= nnet2.maxes))
        nnetEval2 = nnet2.evaluate_network(testInput)

        percChangePB = np.max(np.abs((nnetEval - tf_eval) / nnetEval)) * 100.0
        percChangeNNet = np.max(np.abs((nnetEval - nnetEval2) / nnetEval)) * 100.0

        # Evaluation should not change
        self.assertTrue(percChangePB < 1e-3)
        self.assertTrue(percChangeNNet < 1e-3)


if __name__ == '__main__':
    unittest.main()
