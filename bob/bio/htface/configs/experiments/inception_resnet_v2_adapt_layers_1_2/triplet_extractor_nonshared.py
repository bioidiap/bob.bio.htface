#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.io.base
import bob.bio.face
from bob.bio.htface.extractor import TripletEmbedding
import os

import bob.ip.tensorflow_extractor
import tensorflow as tf
from bob.bio.htface.architectures import inception_resnet_v2_adapt_layers_1_2_head
from bob.bio.htface.utils import get_cnn_model_name

# UPDATE YOUR NAMES HERE
architecture = inception_resnet_v2_adapt_layers_1_2_head
model_name = "triplet_inceptionv2_layers_1_2_nonshared"


# The model filename depends on the database and its protocol and those values are
# chain loaded via database.py
model_filename = get_cnn_model_name(temp_dir, model_name,
                                    database.name, protocol)
#bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding)

#from bob.bio.htface.extractor import TensorflowEmbedding
#extractor = TensorflowEmbedding(bob.ip.tensorflow_extractor.Extractor(model_filename, inputs, embedding))


extractor = TripletEmbedding(model_filename, architecture, shape=(1, 160, 160, 1))

