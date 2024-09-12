import onnx
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from onnx_tf.backend import prepare

# Load the ONNX model
# onnx_model = onnx.load("weights/best.onnx")

# Convert ONNX model to TensorFlow
tf_rep = prepare('/Users/administrator/Documents/python_projects/traffic-gesture-model/human-pose/weights/best.onnx')
print(tf_rep)
# Export the TensorFlow model
tf_rep.export_graph("saved_model")
