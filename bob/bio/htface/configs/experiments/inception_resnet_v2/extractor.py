#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import TensorflowEmbedding


import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2


### Preparing the input of the extractor ####
model_filename = inception_resnet_v2_casia_webface_gray # Value wired by path


inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits,_ = inception_resnet_v2(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                  mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")

extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))

