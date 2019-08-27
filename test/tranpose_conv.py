import tensorflow as tf
from tensorflow.contrib import slim

image_input = tf.constant(value=1, dtype=tf.float32, shape=(1, 7, 7, 3), name="image_input")
transpose_feature1 = slim.conv2d_transpose(image_input, num_outputs=1, kernel_size=(3, 3), stride=1, padding="SAME",
                                           weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))
transpose_feature2 = slim.conv2d_transpose(image_input, num_outputs=1, kernel_size=(3, 3), stride=2, padding="SAME",
                                           weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))
transpose_feature3 = slim.conv2d_transpose(image_input, num_outputs=1, kernel_size=(3, 3), stride=3, padding="SAME",
                                           weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))

image_input_v = tf.constant(value=1, dtype=tf.float32, shape=(1, 8, 8, 3), name="image_input")
transpose_feature1v = slim.conv2d_transpose(image_input_v, num_outputs=1, kernel_size=(3, 3), stride=1, padding="VALID",
                                            weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))
transpose_feature2v = slim.conv2d_transpose(image_input_v, num_outputs=1, kernel_size=(3, 3), stride=2, padding="VALID",
                                            weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))
transpose_feature3v = slim.conv2d_transpose(image_input_v, num_outputs=1, kernel_size=(3, 3), stride=3, padding="VALID",
                                            weights_initializer=tf.constant_initializer(value=1, dtype=tf.float32))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

transpose_feature1_, transpose_feature2_, transpose_feature3_, \
transpose_feature1v_, transpose_feature2v_, transpose_feature3v_ = sess.run(
    [transpose_feature1, transpose_feature2, transpose_feature3, transpose_feature1v, transpose_feature2v,
     transpose_feature3v])

print(transpose_feature1_.shape)
print(transpose_feature2_.shape)
print(transpose_feature3_.shape)
print(transpose_feature1v_.shape)
print(transpose_feature2v_.shape)
print(transpose_feature3v_.shape)
