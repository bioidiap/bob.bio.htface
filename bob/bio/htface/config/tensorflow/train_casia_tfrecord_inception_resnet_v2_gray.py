import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord
from bob.learn.tensorflow.network import Embedding, LightCNN9, inception_resnet_v2
from bob.learn.tensorflow.loss import BaseLoss, MeanSoftMaxLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg
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



def bob2skimage(bob_image):
    """
    Convert bob color image to the skcit image
    """

    skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))

    for i in range(bob_image.shape[0]):
        skimage[:, :, i] = bob_image[i, :, :]

    return skimage
        


def architecture(inputs):
    graph = inception_resnet_v2(inputs)[0]

    # Adding logits
    graph = slim.fully_connected(graph, 10575, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(0.1),
                scope='Logits', reuse=False)

    return graph

        
# Creating the tf record
tfrecords_filename = "/idiap/temp/tpereira/casia_224x224.tfrecords"
mean_224x224 = "/idiap/temp/tpereira/means_casia224x224.hdf5"
filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100000, name="input")
#filename_queue = None


# Creating the CNN using the TFRecord as input
train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                batch_size=batch_size,
                                input_shape=[None, 224, 224, 3])
   
# Mean subtraction                             
inputs = train_data_shuffler("data")                                    
avg_img = bob2skimage(bob.io.base.load(mean_224x224))
inputs = inputs - avg_img

# Reshaping 128x128
inputs = tf.image.resize_images(inputs, size=[160, 160])
inputs = tf.image.rgb_to_grayscale(inputs, name="rgb_to_gray")    


# Setting the placeholders
# Loss for the softmax
loss = MeanSoftMaxLoss()

trainer = Trainer

# One graph trainer
#trainer = Trainer(train_data_shuffler,
#                 iterations=iterations,
#                  analizer=None,
#                  temp_dir=directory)

