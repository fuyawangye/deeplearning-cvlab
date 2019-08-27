import tensorflow as tf
from net.resnet import utils
import numpy as np
from net.resnet import resnet_v2
import tensorflow.contrib.slim as slim


class Args(object):
    def __init__(self):
        self.stride = 3


args = Args()

inputs = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name="input")
out = resnet_v2.resnet_v2_50(inputs)

print("all variables")
for i in tf.all_variables():
    print(i.name)

print("train variables")
for i in tf.trainable_variables():
    print(i.name)

saver = tf.train.Saver(var_list=tf.trainable_variables())

with tf.Session().as_default() as sess:
    tf.global_variables_initializer().run()
    sess.run(out, feed_dict={inputs: np.ones((64, 224, 224, 3), dtype=np.float)})
    saver.save(sess, "./model/model", global_step=1)