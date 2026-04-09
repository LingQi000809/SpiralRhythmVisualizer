import tensorflow as tf

with tf.io.gfile.GFile("public/models/genre519/model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    print(node.name)


# Convert PB to JSON in terminal:
# tensorflowjs_converter \    
# --input_format=tf_frozen_model  \
# --output_node_names="model/Softmax"  \ # change to the last line found in the above code's output
# public/models/moods_mirex-mtt-musicnn/model.pb  \
# public/models/moods_mirex-mtt-musicnn/