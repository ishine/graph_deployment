"""
https://developers.googleblog.com/2018/03/tensorrt-integration-with-tensorflow.html
"""
import sys

frozen_graph_file,trt_graph_file=sys.argv[1:]

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
with tf.Session() as sess:
    # First deserialize your frozen graph:
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    # Now you can create a TensorRT inference graph from your
    # frozen graph:
    converter = trt.TrtGraphConverter(
	    input_graph_def=frozen_graph,
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_batch_size=5,
	    nodes_blacklist=["add:0"]) #output nodes

    trt_graph = converter.convert()
    # Import the TensorRT graph into a new graph and run:
    """
    output_node = tf.import_graph_def(
        trt_graph,
        return_elements=["add:0"])
    """

with tf.gfile.GFile(trt_graph_file, "wb") as f:
    f.write(trt_graph.SerializeToString())
    print("%d ops in the final graph." % len(trt_graph.node))