from tensorflow.python import pywrap_tensorflow

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# pai
flags.DEFINE_string("checkpoint", None, "")
flags.DEFINE_string("name", None, "")


def main(_):
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)

    tensor = reader.get_tensor(FLAGS.name)
    print type(tensor)
    for i in tensor:
        print (i)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
