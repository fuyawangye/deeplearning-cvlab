import tensorflow as tf




saver = tf.train.import_meta_graph("./model/model-1.meta")
# saver.recover_last_checkpoints("./model/model-1")
with tf.compat.v1.Session() as sess:
    saver.restore(sess, './model/model-1')
    mean = tf.get_collection("resnet_v2_50/block1/unit_1/bottleneck_v2/preact/moving_mean:0")
    print(mean)
    mean = tf.get_collection("resnet_v2_50/block1/unit_1/bottleneck_v2/preact/moving_mean")[0]
    print(mean)
    mean = tf.get_collection("resnet_v2_50/conv1/weights:0")
    print(mean)
    mean = tf.get_collection("resnet_v2_50/conv1/weights")[0]
    print(mean)

