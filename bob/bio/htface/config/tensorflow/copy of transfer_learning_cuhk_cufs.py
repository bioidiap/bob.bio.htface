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


def bob2skimage(bob_image):
    """
    Convert bob color image to the skcit image
    """

    skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))

    for i in range(bob_image.shape[0]):
        skimage[:, :, i] = bob_image[i, :, :]

    return skimage
        

def dump_variables_to_dict(trainer):
    """
    Move to trainer
    """

    variables = dict()
    for v in tf.trainable_variables():
        variables[v.name] = v.eval(session=trainer.session)

    return variables


def import_from_dict(dictionary, session):
    """
    Move to trainer
    """

    def get_tfvariable_by_name(name):
        
        for v in tf.trainable_variables():
            
            if str(v.name) == str(name):
                return v
                 
        #raise ValueError("Variable {0} not found".format(v.name))
        return None
        
    variables = dict()
    
    for v in dictionary:
        tf_variable = get_tfvariable_by_name(v)
        if tf_variable is not None:
            session.run(tf_variable.assign(dictionary[v]))

    return variables


def create_network():
        
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
                                            input_shape=[None, 224, 224, 1])
    # Loading the pretrained model
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)
    trainer.create_network_from_file(checkpoint_filename)
    #trainer.create_network_from_file(checkpoint_filename)
    
    import ipdb; ipdb.set_trace();

    variables = dump_variables_to_dict(trainer)
    del trainer


    avg_img = bob2skimage(bob.io.base.load(mean_224x224))

    ### Starting the siamese trainer
    def mean_subtract(input_ph):

        # Mean subtraction                             
        input_ph = input_ph - avg_img
    
        # Reshaping 128x128
        input_ph = tf.image.resize_images(input_ph, size=[160, 160])
        input_ph = tf.image.rgb_to_grayscale(input_ph, name="rgb_to_gray")    

        return input_ph

    input_left = mean_subtract(train_data_shuffler("data")['left'])
    input_right = mean_subtract(train_data_shuffler("data")['right'])

    # Creating the siamese net tha HOOKS with the pretrained data
    #network = LightCNN9(n_classes=10575)
    #graph = dict()

    slim = tf.contrib.slim

    #with tf.device(self.device):        
    #inputs = train_data_shuffler("data")['left']
    #graph = slim.conv2d(inputs, 96, [5, 5], activation_fn=tf.nn.relu, stride=1, scope='Conv1', reuse=True)

    loss = ContrastiveLoss(contrastive_margin=0.2)
    siamese_trainer = SiameseTrainer(train_data_shuffler,
                                     iterations=iterations,
                                     analizer=None,
                                     temp_dir=directory)

    graph['left'] = network(input_left, reuse=False, get_class_layer=False)
    graph['right'] = network(input_right, reuse=True, get_class_layer=False)


    learning_rate = constant(0.01, name="regular_lr")
    siamese_trainer.create_network_from_scratch(graph=graph,
                                                loss=loss,
                                                learning_rate=learning_rate,
                                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                                )

    import_from_dict(variables, siamese_trainer.session)
    del variables                                    

    siamese_trainer.train()
    #os.remove(tfrecords_filename)


create_network()

