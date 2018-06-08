#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.ip.tensorflow_extractor
from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm
import tensorflow as tf
from bob.extension import rc
from bob.bio.face_ongoing.extractor import TensorflowEmbedding
model_filename = "/idiap/temp/tpereira/HTFace/cnn/siamese_inceptionv1_adapt_1_4_nonshared_batch_norm/pola_thermal/VIS-thermal-overall-split1/"

#########
# Extraction
#########
inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits, end_points = inception_resnet_v2_batch_norm(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]), mode=tf.estimator.ModeKeys.PREDICT)

#embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")

extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, end_points["Conv2d_1a_3x3"]))

