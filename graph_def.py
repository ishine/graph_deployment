import tensorflow as tf
import sys

gf=tf.GraphDef()
gf.ParseFromString(open(sys.argv[1],"rb").read())
print(gf.node)