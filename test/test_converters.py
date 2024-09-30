import os
import tensorflow as tf
import numpy as np
from NNet.converters.nnet2pb import nnet2pb
from NNet.converters.pb2nnet import pb2nnet
from NNet.python.nnet import NNet
import unittest

class TestConverters(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"  # Ensure this path is correct

    def test_pb(self):
        """Test NNet to TensorFlow Protocol Buffer (PB) conversion and back."""
        pbFile = self.nnetFile.replace(".nnet", ".pb")  # Fixed double dot issue
        nnetFile2 = self.nnetFile.replace(".nnet", "v2.nnet")

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

        # Define session to evaluate TensorFlow model
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")
            input_tensor = sess.graph.get_tensor_by_name("Placeholder:0")  # Ensure correct placeholder name
            output_tensor = sess.graph.get_tensor_by_name("Identity:0")  # Ensure correct output name

            # Run the session
            testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
            tf_output = sess.run(output_tensor, feed_dict={input_tensor: testInput.reshape(1, -1)})

        # Evaluate NNet models
        nnetEval = nnet.evaluate_network(testInput)
        nnetEval2 = nnet2.evaluate_network(testInput)

        # Check TensorFlow evaluation vs NNet evaluation
        np.testing.assert_allclose(tf_output.flatten(), nnetEval, rtol=1e-5, err_msg="TensorFlow output mismatch.")
        np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5, err_msg="NNet model mismatch after conversion.")

if __name__ == '__main__':
    unittest.main()
