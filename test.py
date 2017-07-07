from bob.bio.htface.datashuffler import SiameseDiskHTFace
from bob.learn.tensorflow.loss import ContrastiveLoss
from bob.learn.tensorflow.trainers import SiameseTrainer, constant
from bob.learn.tensorflow.network import Chopra
import tensorflow as tf

directory = "./temp/inception"

def create_architecture(placeholder):
    initializer = tf.contrib.layers.xavier_initializer(seed=10)  # Weights initializer

    slim = tf.contrib.slim
    graph = slim.conv2d(placeholder, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer)
    graph = slim.flatten(graph, scope='flatten1')
    graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1', weights_initializer=initializer)

    return graph

import ipdb; ipdb.set_trace()

# Loading data
from bob.db.cuhk_cufs.query import Database
database = Database(original_directory="/Users/tiago.pereira/Documents/database/cuhk_cufs_process",
                    original_extension=".png",
                    arface_directory="", xm2vts_directory="")

train_data_shuffler = SiameseDiskHTFace(database=database, protocol="cuhk_p2s",
                                        batch_size=8,
                                        input_shape=[None, 224, 224, 1])

# Loss for the softmax
loss = ContrastiveLoss()

# Creating inception model
inputs = train_data_shuffler("data", from_queue=False)

from tensorflow.contrib.slim.python.slim.nets import inception
graph = dict()
chopra = Chopra()
graph['left'] = chopra(inputs['left'])
graph['right'] = chopra(inputs['right'], reuse=True)

#graph['left'] = inception.inception_v1(inputs['left'])[0]
#graph['right'] = inception.inception_v1(inputs['right'], reuse=True)[0]


# One graph trainer
iterations = 100
trainer = SiameseTrainer(train_data_shuffler,
                         iterations=iterations,
                         analizer=None,
                         temp_dir=directory
                         )
trainer.create_network_from_scratch(graph=graph,
                                    loss=loss,
                                    learning_rate=constant(0.01, name="regular_lr"),
                                    optimizer=tf.train.GradientDescentOptimizer(0.01)
                                    )
trainer.train()
x = 0