import unittest
import os
import tensorflow as tf
from io import StringIO
from unittest.mock import patch
from NNet.converters.nnet2pb import nnet2pb


class TestNNet2PB(unittest.TestCase):

    def setUp(self):
        self.nnetFile = "nnet/TestNetwork.nnet"
        self.assertTrue(os.path.exists(self.nnetFile), f"{self.nnetFile} not found!")

    def tearDown(self):
        for ext in [".pb"]:
            file = self.nnetFile.replace(".nnet", ext)
            if os.path.exists(file):
                os.remove(file)

    def test_default_pb_filename(self):
        """Test default behavior when no pbFile is provided."""
        nnet2pb(self.nnetFile)  # No pbFile specified
        default_pb_file = self.nnetFile.replace(".nnet", ".pb")
        self.assertTrue(os.path.exists(default_pb_file), f"Default PB file {default_pb_file} not created!")
        os.remove(default_pb_file)  # Cleanup

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

    @patch("sys.argv", ["nnet2pb.py", "nnet/TestNetwork.nnet"])
    def test_command_line_valid(self):
        """Test command-line execution with valid arguments."""
        from NNet.converters.nnet2pb import main
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
        default_pb_file = self.nnetFile.replace(".nnet", ".pb")
        self.assertTrue(os.path.exists(default_pb_file), "Default PB file not created via argparse!")
        os.remove(default_pb_file)  # Cleanup

    @patch("sys.argv", ["nnet2pb.py"])
    def test_command_line_missing_args(self):
        """Test command-line execution with missing arguments."""
        from NNet.converters.nnet2pb import main
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as excinfo:
                main()
            self.assertEqual(excinfo.exception.code, 1)  # Ensure SystemExit
        output = mock_stdout.getvalue()
        self.assertIn("Usage: python nnet2pb.py <nnetFile> [pbFile] [output_node_names]", output)


if __name__ == "__main__":
    unittest.main()
