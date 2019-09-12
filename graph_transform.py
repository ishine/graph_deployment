from tensorflow.tools.graph_transforms import TransformGraph
import tensorflow as tf
import os

def get_graph_def_from_file(graph_filepath):
  with tf.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms, output_node):
  input_names = []
  output_names = [output_node]

  graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))

  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,
      output_names,
      transforms)

  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name='optimized_model.pb')
  print('Graph optimized!')

import sys

model_dir,graph_filename=sys.argv[1:]


transforms = [
 'remove_nodes(op=Identity)', 
 'merge_duplicate_nodes',
 'strip_unused_nodes',
 'fold_constants(ignore_errors=true)',
 'fold_batch_norms'
]

transforms = ['quantize_nodes', 'quantize_weights']

transforms = ['round_weights(num_steps=256)']

transforms = [ 'quantize_weights']

optimize_graph(model_dir,graph_filename,transforms,output_node="prefix/add:0")
