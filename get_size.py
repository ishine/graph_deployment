import tensorflow as tf
import os
def get_size(model_dir, model_file='saved_model.pb'):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if os.path.exists(
      os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(
        model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
  print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))
get_size("checkpoint_full/saved_strip")

from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter.trt.TrtGraphConverter(input_graph_def=frozen_graph)
frozen_graph=converter.convert