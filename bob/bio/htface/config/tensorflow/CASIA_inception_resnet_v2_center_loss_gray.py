import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecordImage
from bob.learn.tensorflow.network import Embedding, LightCNN9, inception_resnet_v2
from bob.learn.tensorflow.loss import mean_cross_entropy_loss, mean_cross_entropy_center_loss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg
from bob.learn.tensorflow.network.utils import append_logits
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
num_epochs=60
n_classes = 10575


#tf_record_path = "/idiap/temp/tpereira/databases/casia_webface/182x/tfrecord/"
#tf_record_path_validation = "/idiap/temp/tpereira/databases/LFW/182x/tfrecord/"
tf_record_path = "/idiap/temp/tpereira/databases/casia_webface/182x/tfrecord_onlygood/"
tf_record_path_validation = "/idiap/temp/tpereira/databases/LFW/182x/tfrecord_onlygood/"



def build_graph(inputs, reuse=False):

    prelogits = inception_resnet_v2(inputs, reuse=reuse)[0]
    
    return prelogits 

# Creating the tf record
tfrecords_filename = [os.path.join(tf_record_path, f) for f in os.listdir(tf_record_path)]
filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=num_epochs, name="input")

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
                                gray_scale=True
                                )
                                
validation_data_shuffler  = TFRecordImage(filename_queue=filename_queue_validation,
                                batch_size=validation_batch_size,
                                input_shape=[None, 182, 182, 3],
                                output_shape=[None, 160, 160, 3],
                                shuffle=False,
                                input_dtype=tf.uint8,
                                normalization=True,
                                gray_scale=True)

# Tensor used for training
prelogits = build_graph(train_data_shuffler("data", from_queue=False))
labels = train_data_shuffler("label", from_queue=False)
logits = append_logits(prelogits, n_classes=n_classes)

# Tensor used for validation
validation_graph = tf.nn.l2_normalize(build_graph(validation_data_shuffler("data", from_queue=False), reuse=True), 1)
 
# POINTER TO THE ARCHITECTURE

# Setting the placeholders
#loss = MeanSoftMaxLossCenterLoss(n_classes=n_classes)
# Setup from (https://github.com/davidsandberg/facenet/issues/391)
#loss = MeanSoftMaxLossCenterLoss(alpha=0.5, factor=0.02, n_classes=n_classes)
loss = mean_cross_entropy_center_loss(logits, prelogits, labels, alpha=0.5, factor=0.02, n_classes=n_classes)


### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.01)


### SOLVER ###
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
optimizer = tf.train.AdagradOptimizer(learning_rate)

validate_with_embeddings = True

trainer = Trainer

