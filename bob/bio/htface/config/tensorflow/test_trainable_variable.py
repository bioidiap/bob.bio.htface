import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord
from bob.learn.tensorflow.network import Embedding, LightCNN9
from bob.learn.tensorflow.loss import MeanSoftMaxLoss, ContrastiveLoss
from bob.learn.tensorflow.trainers import Trainer, constant, SiameseTrainer
from bob.learn.tensorflow.utils import load_mnist
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg
from bob.bio.htface.datashuffler import SiameseDiskHTFace
import bob.io.base
import bob.io.image
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


directory = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/resnet_gray/"
checkpoint_filename = "/idiap/temp/tpereira/casia_webface/inception_resnet_v2_gray/model_snapshot707500.ckp-707500.meta"
mean_224x224 = "/idiap/temp/tpereira/means_casia224x224.hdf5"


def create_network():
        
    # Creating the tf record
    #filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100000, name="input")
    #filename_queue = None
    
    




create_network()

