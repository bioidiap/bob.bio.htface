#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import TensorflowEmbedding


import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm


# Preparing the input of the extractor ####
model_filename = inception_resnet_v2_casia_webface_rgb_batch_norm # Value wired by path


inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 3))

# Taking the embedding
prelogits, end_points = inception_resnet_v2_batch_norm(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                                       mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")
extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding, debug=False))

