import numpy
import sys
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord
from bob.learn.tensorflow.network import Embedding, LightCNN9, inception_resnet_v2, Chopra
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


directory = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/resnet_gray_256-to-128/"
#checkpoint_filename = "/idiap/temp/tpereira/casia_webface/inception_resnet_v2_gray/model_snapshot707500.ckp-707500.meta"
checkpoint_filename = "/idiap/temp/tpereira/casia_webface/official_checkpoints/resnet_inception_v2_gray/"
mean_224x224 = "/idiap/temp/tpereira/means_casia224x224.hdf5"


def bob2skimage(bob_image):
    """
    Convert bob color image to the skcit image
    """

    skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))

    #for i in range(bob_image.shape[0]):
    skimage[:, :, 2] = bob_image[0, :, :]
    skimage[:, :, 1] = bob_image[1, :, :]
    skimage[:, :, 0] = bob_image[2, :, :]

    return skimage
    

def build_graph(inputs, get_embeddings=False, reuse=False):

    avg_img = bob2skimage(bob.io.base.load(mean_224x224))
    inputs = inputs - avg_img

    inputs = tf.image.resize_images(inputs, size=[160, 160])
    inputs = tf.image.rgb_to_grayscale(inputs, name="rgb_to_gray")    

    prelogits, end_points = inception_resnet_v2(inputs, reuse=reuse, is_training=False)
    #prelogits = Chopra()(inputs, reuse=reuse, end_point="fc1")

    # Adding logits
    if get_embeddings:
        #embeddings = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")
        return prelogits, end_points
    else:
        logits = slim.fully_connected(prelogits, 99197, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(0.2),
                    scope='Logits', reuse=reuse)

        #logits_prelogits = dict()
        #logits_prelogits['logits'] = logits
        #logits_prelogits['prelogits'] = prelogits

        return logits, end_points


def transfer_graph(inputs, get_embeddings=False, reuse=False):
    prelogits = inputs
    with tf.variable_scope('Transfer'):
        prelogits = slim.fully_connected(prelogits, 256, activation_fn=tf.nn.relu, 
                                         scope='NonLinearity', reuse=reuse)

        prelogits = slim.batch_norm(prelogits, scope="covariate_shift", reuse=reuse)

        prelogits = slim.fully_connected(prelogits, 128, activation_fn=None, 
                                         scope='Bottleneck', reuse=reuse)

    return prelogits 
    
# Creating the tf record
#filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100000, name="input")
#filename_queue = None

# Creating the CNN using the TFRecord as input
from bob.db.cuhk_cufs.query import Database
database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/SIAMESE/split1/preprocessed/",
                    original_extension=".hdf5",
                    arface_directory="", xm2vts_directory="")

train_data_shuffler = SiameseDiskHTFace(database=database, protocol="search_split1_p2s",
                                        batch_size=8,
                                        input_shape=[None, 224, 224, 3])

# Loading the pretrained model
trainer = SiameseTrainer(train_data_shuffler,
                          iterations=iterations,
                          analizer=None,
                          temp_dir=directory)

# Loss for the Siamese
loss = ContrastiveLoss(contrastive_margin=2.)
learning_rate=constant(0.1, name="regular_lr")

optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#optimizer=tf.train.GradientDescentOptimizer(0.01)

# Preparing the graph                              

input_pl = train_data_shuffler("data")
graph = dict()
graph['left'], end_points = build_graph(input_pl['left'], get_embeddings=True)
graph['left'] = transfer_graph(graph['left'])

graph['right'], _ = build_graph(input_pl['right'], reuse=True, get_embeddings=True)
graph['right'] = transfer_graph(graph['right'], reuse=True)
  

# Bootstraping the siamease                        
trainer.create_network_from_scratch(graph=graph,
                                    loss=loss,
                                    learning_rate=learning_rate,
                                    optimizer=optimizer)
 
# Loading the pretrained variables
var_list = ["InceptionResnetV2"]
trainer.load_variables_from_external_model(checkpoint_filename, var_list=var_list)

trainer.train()
#var
# model_snapshot16500.ckp


