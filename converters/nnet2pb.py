import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from NNet.utils.readNNet import readNNet
from NNet.utils.normalizeNNet import normalizeNNet

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def nnet2pb(
    nnetFile: str, 
    pbFile: str = "", 
    output_node_names: str = "y_out", 
    normalizeNetwork: bool = False
) -> None:
    """
    Convert a .nnet file to a frozen TensorFlow graph (.pb file).

    Args:
        nnetFile (str): Path to the .nnet file to convert.
        pbFile (str, optional): Name for the .pb file. Defaults to same as the input file name.
        output_node_names (str, optional): Name of the output operation. Defaults to 'y_out'.
        normalizeNetwork (bool): Normalize weights and biases if True. Defaults to False.
    """
    # Read network weights and biases
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)

    inputSize = weights[0].shape[1]

    # Default pb filename if not provided
    if not pbFile:
        pbFile = f"{nnetFile[:-5]}.pb"

    class NNetModel(tf.Module):
        def __init__(self, weights, biases):
            super().__init__()
            self.weights = [tf.Variable(w.T, dtype=tf.float32) for w in weights]
            self.biases = [tf.Variable(b, dtype=tf.float32) for b in biases]

        @tf.function(input_signature=[tf.TensorSpec([None, inputSize], tf.float32)])
        def __call__(self, x):
            # Build the model layer by layer
            for i in range(len(self.weights) - 1):
                x = tf.nn.relu(tf.matmul(x, self.weights[i]) + self.biases[i])
            return tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1], name=output_node_names)

    # Create and save the model
    model = NNetModel(weights, biases)
    concrete_func = model.__call__.get_concrete_function()

    # Convert to a frozen graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_graph_def = frozen_func.graph.as_graph_def()

'''
    Given a session with a graph loaded, save only the variables needed for evaluation to a .pb file
    
    Args:
        sess (tf.session): Tensorflow session where graph is defined
        output_graph_name (str): Name of file for writing frozen graph
        output_node_names (str): Name of the output operation in the graph, comma separated if there are multiple output operations
    '''
    
    # Save the frozen graph to a .pb file
    tf.io.write_graph(frozen_graph_def, ".", pbFile, as_text=False)
    print(f"Saved frozen graph to {pbFile}")

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python nnet2pb.py <nnetFile> [pbFile] [output_node_names]")
        sys.exit(1)

    nnetFile = sys.argv[1]
    pbFile = sys.argv[2] if len(sys.argv) > 2 else ""
    output_node_names = sys.argv[3] if len(sys.argv) > 3 else "y_out"

    # Call the conversion function
    nnet2pb(nnetFile, pbFile, output_node_names)

if __name__ == "__main__":
    main()
