from NNet.converters.nnet2pb import nnet2pb

# Define the paths for the NNet file and the TensorFlow PB file
nnet_file = "../nnet/TestNetwork.nnet"
pb_file = "../nnet/TestNetwork2.pb"

# Try to convert the NNet file to TensorFlow PB format, with error handling
try:
    print(f"Converting {nnet_file} to {pb_file}...")
    nnet2pb(nnet_file, pb_file)
    print(f"Successfully converted {nnet_file} to {pb_file}")
except Exception as e:
    print(f"Failed to convert {nnet_file} to {pb_file}. Error: {e}")
