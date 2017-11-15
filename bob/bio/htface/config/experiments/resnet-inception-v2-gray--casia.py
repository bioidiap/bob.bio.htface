#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.face
import bob.bio.htface.algorithm
import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import TensorflowEmbedding


import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2


### Preparing the input of the extractor ####
model_filename = "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/"
inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

# Taking the embedding
prelogits,_ = inception_resnet_v2(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                  mode=tf.estimator.ModeKeys.PREDICT)
embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")


# This is the size of the image that this model expects
CROPPED_IMAGE_HEIGHT = 160
CROPPED_IMAGE_WIDTH = 160

# eye positions for frontal images
RIGHT_EYE_POS = (48, 53)
LEFT_EYE_POS = (48, 107)

# Detects the face and crops it without eye detection
preprocessor = bob.bio.face.preprocessor.FaceCrop(
  cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
  cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
  color_channel='gray'
)

extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))


algorithm = 'distance-cosine'

