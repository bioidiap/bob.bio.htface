import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord
from bob.learn.tensorflow.network import Embedding, LightCNN9
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
iterations = 2000000
seed = 10
slim = tf.contrib.slim


directory = "/idiap/temp/tpereira/casia_webface/lightcnn_9_maxout_0to1"
tfrecords_filename = "/idiap/temp/tpereira/casia_224x224.tfrecords"
#mean_224x224 = "/idiap/temp/tpereira/means_casia224x224.hdf5"


def create_network():
        
    # Creating the tf record
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100000, name="input")
    #filename_queue = None
    

    # Creating the CNN using the TFRecord as input
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size,
                                    input_shape=[None, 224, 224, 3])
       
    # Mean subtraction                             
    inputs = train_data_shuffler("data")                                    
    
    # Reshaping 128x128
    inputs = tf.image.resize_images(inputs, size=[128, 128])
    inputs = tf.image.rgb_to_grayscale(inputs, name="rgb_to_gray")    

    # Normalizing from 0 to 1
    inputs = inputs * 0.00390625

    network = LightCNN9(n_classes=10575)
    graph = network(inputs)

    # Setting the placeholders
    # Loss for the softmax
    loss = MeanSoftMaxLoss()

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                     iterations=iterations,
                      analizer=None,
                      temp_dir=directory)
    #trainer.create_network_from_file("/idiap/temp/tpereira/casia_webface/lightcnn_9_maxout/model_snapshot1303000.ckp-1303000.meta")

    learning_rate = constant(0.01, name="regular_lr")
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )

    trainer.train()
    #os.remove(tfrecords_filename)


create_network()

