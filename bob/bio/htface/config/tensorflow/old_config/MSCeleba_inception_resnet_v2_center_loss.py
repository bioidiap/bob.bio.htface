import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecordImage
from bob.learn.tensorflow.network import Embedding, LightCNN9, inception_resnet_v2
from bob.learn.tensorflow.loss import BaseLoss, MeanSoftMaxLoss, MeanSoftMaxLossCenterLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
import bob.io.base
import bob.io.image
import tensorflow as tf
import shutil
import os

batch_size = 16
validation_batch_size = 400
iterations = 20000000
seed = 10
slim = tf.contrib.slim
final_size = 160

tf.set_random_seed(seed)

tf_record_path = "/idiap/temp/tpereira/databases/MSCeleba/tfrecord_182x/"
tf_record_path_validation = "/idiap/temp/tpereira/databases/LFW/182x/tfrecord/"
n_classes = 99197

def build_graph(inputs, get_embeddings=False, reuse=False, final_size=160):

    prelogits = inception_resnet_v2(inputs, reuse=reuse)[0]
    #graph = inception_resnet_v1(inputs, reuse=reuse)[0]

    # Adding logits
    if get_embeddings:
        embeddings = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")
        return embeddings
    else:
        logits = slim.fully_connected(prelogits, n_classes, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    scope='Logits', reuse=reuse)

        return logits, prelogits

# Creating the tf record
tfrecords_filename = [os.path.join(tf_record_path, f) for f in os.listdir(tf_record_path)]
filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=100000, name="input")

tfrecords_filename_validation = [os.path.join(tf_record_path_validation, f) for f in os.listdir(tf_record_path_validation)]
filename_queue_validation = tf.train.string_input_producer(tfrecords_filename_validation , num_epochs=10000000, name="input_validation")


## Files are saved in 182x182
# Creating the CNN using the TFRecord as input
train_data_shuffler  = TFRecordImage(filename_queue=filename_queue,
                                batch_size=batch_size,
                                input_shape=[None, 182, 182, 3],
                                output_shape=[None, 160, 160, 3],
                                input_dtype=tf.uint8,
                                normalization=True,
                                gray_scale=False
                                )
                                
validation_data_shuffler  = TFRecordImage(filename_queue=filename_queue_validation,
                                batch_size=validation_batch_size,
                                input_shape=[None, 182, 182, 3],
                                output_shape=[None, 160, 160, 3],
                                shuffle=False,
                                input_dtype=tf.uint8,
                                normalization=True,
                                gray_scale=False)

# Tensor used for training
train_graph, prelogits = build_graph(train_data_shuffler("data", from_queue=False), final_size=final_size)

# Tensor used for validation
validation_graph = build_graph(validation_data_shuffler("data", from_queue=False), get_embeddings=True, reuse=True, final_size=final_size)
 
# POINTER TO THE ARCHITECTURE

# Setting the placeholders
loss = MeanSoftMaxLossCenterLoss(n_classes=n_classes, factor=0.01)
#loss = MeanSoftMaxLoss()


### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.01)


### SOLVER ###
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
optimizer = tf.train.AdagradOptimizer(learning_rate)

validate_with_embeddings = True

trainer = Trainer

