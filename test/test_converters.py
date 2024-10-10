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
        
        try:
            inputTensor = sess.graph.get_tensor_by_name("input:0")  # Adjust name if needed
            outputTensor = sess.graph.get_tensor_by_name("y_out:0")  # Adjust name if needed
        except KeyError as e:
            self.fail(f"Tensor not found in graph: {e}")

        testInput = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32).reshape(1, -1)

        try:
            pbEval = sess.run(outputTensor, feed_dict={inputTensor: testInput})[0]
        except Exception as e:
            self.fail(f"Failed to run TensorFlow inference: {e}")

    # NNet evaluations
    nnetEval = nnet.evaluate_network(testInput[0])  # NNet expects 1D array
    nnetEval2 = nnet2.evaluate_network(testInput[0])

    np.testing.assert_allclose(nnetEval, pbEval.flatten(), rtol=1e-5)
    np.testing.assert_allclose(nnetEval, nnetEval2, rtol=1e-5)
