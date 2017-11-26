from bob.learn.tensorflow.network import inception_resnet_v2
from bob.learn.tensorflow.estimators import Siamese
#from bob.learn.tensorflow.dataset.tfrecords import batch_data_and_labels_image_augmentation, shuffle_data_and_labels_image_augmentation
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import contrastive_loss
import os
import tensorflow as tf

from bob.bio.htface.dataset.siamese_htface import shuffle_data_and_labels_image_augmentation

from bob.learn.tensorflow.utils import reproducible


learning_rate = 0.01
#data_shape = (182, 182, 3)  # size of atnt images
#output_shape = (160, 160)

data_shape = (160, 160, 1)  # size of atnt images
output_shape = None
data_type = tf.uint8

batch_size = 16
validation_batch_size = 250
epochs = 100
embedding_validation = True

model_dir = "/idiap/temp/tpereira/casia_webface/cuhk_cufs/siamese-transfer-idiap_inception_v2_gray--casia"
protocol = "search_split1_p2s"

# Model for transfer
extra_checkpoint = {"checkpoint_path": "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/", 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                    "is_trainable": True
                   }

# Setting the database
from bob.db.cuhk_cufs.query import Database
database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/idiap_inception_v2_gray--casia/split1/preprocessed/",
                    original_extension=".hdf5")


def train_input_fn():
    return shuffle_data_and_labels_image_augmentation(database, protocol, data_shape, data_type,
                                                      batch_size, epochs=epochs, buffer_size=10**3,
                                                      gray_scale=False, 
                                                      output_shape=None,
                                                      random_flip=True, random_brightness=False,
                                                      random_contrast=False,
                                                      random_saturation=False,
                                                      per_image_normalization=True, 
                                                      groups="world", purposes="train",
                                                      extension="hdf5")

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_checkpoints_steps=500)

estimator = Siamese(model_dir=model_dir,
                    architecture=inception_resnet_v2,
                    optimizer=tf.train.AdagradOptimizer(learning_rate),
                    validation_batch_size=validation_batch_size,
                    config=run_config,
                    loss_op=contrastive_loss,
                    extra_checkpoint=extra_checkpoint)

hooks = [LoggerHookEstimator(estimator, 16, 1),
         tf.train.SummarySaverHook(save_steps=5,
                                   output_dir=model_dir,
                                   scaffold=tf.train.Scaffold(),
                                   summary_writer=tf.summary.FileWriter(model_dir) )]

