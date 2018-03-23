#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Plot the sum of the fft absolute values for each convolved signal

Usage:
  bob_htface_convolve_and_view.py
  bob_htface_convolve_and_view.py -h | --help

Options:
  --end-points=<arg>                  List of the end points [default: all]
  --demo                              Plot the demo figure
  -h --help                           Show this screen.
"""


import bob.io.base
import bob.io.image

import bob.sp
import os
import numpy
from docopt import docopt
#from bob.learn.tensorflow.network import inception_resnet_v2, inception_resnet_v2_batch_norm
from bob.bio.htface.architectures.inception_v2_batch_norm import inception_resnet_v2_adapt_layers_1_5_head, inception_resnet_v2_adapt_first_head

import tensorflow as tf
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#import numpy
#from .registered_baselines import all_baselines, resources
#import pkg_resources
#from bob.bio.base.utils import read_config_file
#from bob.bio.htface.utils import get_cnn_model_name
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


####### TODO: ADD AS A PARAMETER
#model_dir = "/idiap/temp/tpereira/HTFace/cnn/inception_resnet_v2_adapt_layers_1_5_nonshared_batch_norm/cuhk_cufs/search_split1_p2s/"
#preprocess_dir = "/idiap/temp/tpereira/HTFace/cuhk_cufs/idiap_casia_inception_v2_gray/search_split1_p2s/preprocessed/"

#model_dir = "/idiap/temp/tpereira/HTFace/cnn/idiap_casia_inception_v2_gray_adapt_first_layer_nonshared_batch_norm/CBSR_NIR_VIS_2/view2_1/"
#preprocess_dir = "/idiap/temp/tpereira/HTFace/casia_nir_vis/idiap_casia_inception_v2_gray/view2_1/preprocessed/"


model_dir = "/idiap/temp/tpereira/HTFace/cnn/inception_resnet_v2_adapt_layers_1_5_nonshared_batch_norm/pola_thermal/VIS-polarimetric-overall-split1/"
preprocess_dir = "/idiap/temp/tpereira/HTFace/pola_thermal/idiap_casia_inception_v2_gray/VIS-polarimetric-overall-split1/preprocessed/"


#architecture = inception_resnet_v2_adapt_layers_1_5_head
architecture = inception_resnet_v2_adapt_first_head



def do_fft(image):
    img_complex = image.astype("complex128")
    img_fft = bob.sp.fft(img_complex)
    img_fft = abs(bob.sp.fftshift(img_fft))
    #psd = 10*numpy.log(abs(img_fft)**2)    
    return img_fft


def convolve_db(session, modality, placeholder, end_points, key):

    import bob.db.cuhk_cufs
    import bob.db.cbsr_nir_vis_2
    import bob.db.pola_thermal


    #database = bob.db.cuhk_cufs.Database()
    #objects = database.objects(protocol="search_split1_p2s", groups="world", modality=modality)

    #database = bob.db.cbsr_nir_vis_2.Database()
    #objects = database.objects(protocol="view2_1", groups="world", modality=modality)
    
    database = bob.db.pola_thermal.Database()
    objects = database.objects(protocol="VIS-polarimetric-overall-split1", groups="world", modality=modality)
    
 
    convolved_images = None
    n_samples = 0
    for o in objects:
        path = o.make_path(preprocess_dir, ".hdf5")
        data = numpy.reshape(bob.io.base.load(path).astype("float32"), (1, 160, 160, 1))
        
        if convolved_images is None:
            convolved_images = session.run(end_points[key], feed_dict={placeholder: data})
        else:
            convolved_images += session.run(end_points[key], feed_dict={placeholder: data})
        n_samples += 1

    convolved_images = convolved_images / n_samples
    return convolved_images


def normalize4save(img):
    return (255 * ((img - numpy.min(img)) / (numpy.max(img)-numpy.min(img)))).astype("uint8")


def main():

    args = docopt(__doc__, version='Run experiment')
    set_verbosity_level(logger, 2)
    #output_file_name = args["<output-file-name>"]    
    #database_name = args["<database>"]
    #baselines = ["idiap_casia_inception_v2_gray_adapt_first_layer_nonshared_batch_norm"]

  
    # Loading TF MODEL
    inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

    # Getting the end_points
    _, left_end_points = architecture(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]), mode=tf.estimator.ModeKeys.PREDICT, is_left = True)
    _, right_end_points = architecture(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]), mode=tf.estimator.ModeKeys.PREDICT, is_left = False, reuse=True)


    scopes = {"InceptionResnetV2/Conv2d_1a_3x3_left/weights": "InceptionResnetV2/Conv2d_1a_3x3_left/weights",
              "InceptionResnetV2/Conv2d_1a_3x3_right/weights": "InceptionResnetV2/Conv2d_1a_3x3_right/weights",
                                              
              "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/moving_mean": "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/moving_mean",
              "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/moving_variance": "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/moving_variance",
              "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/beta": "InceptionResnetV2/Conv2d_1a_3x3_left/BatchNorm/beta",

              "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/moving_mean": "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/moving_mean",
              "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/moving_variance": "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/moving_variance",
              "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/beta": "InceptionResnetV2/Conv2d_1a_3x3_right/BatchNorm/beta",
              }
              
    tf.contrib.framework.init_from_checkpoint(model_dir, scopes)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    
    pdf = PdfPages("IMG_THERMAL.pdf")
    
    convolved_images = convolve_db(session, "VIS", inputs, left_end_points, "Conv2d_1a_3x3_left")    
    fig = plt.figure()
    fig.suptitle("VIS")
    for i in range(convolved_images.shape[3]):    
        plt.subplot(8, 4, i+1)
        #plt.imshow(do_fft(convolved_images[0,:,:,i]))
        plt.imshow(convolved_images[0,:,:,i])
        plt.axis('off')
    pdf.savefig(fig)


    convolved_images = convolve_db(session, "THERMAL", inputs, left_end_points, "Conv2d_1a_3x3_left")
    fig = plt.figure()
    fig.suptitle("NIR non adapted")
    for i in range(convolved_images.shape[3]):    
        plt.subplot(8, 4, i+1)
        #plt.imshow(do_fft(convolved_images[0,:,:,i]))
        plt.imshow(convolved_images[0,:,:,i])
        plt.axis('off')
    pdf.savefig(fig)


    convolved_images = convolve_db(session, "THERMAL", inputs, right_end_points, "Conv2d_1a_3x3_right")
    fig = plt.figure()
    fig.suptitle("NIR adapted")
    for i in range(convolved_images.shape[3]):
        plt.subplot(8, 4, i+1)
        #plt.imshow(do_fft(convolved_images[0,:,:,i]))
        plt.imshow(convolved_images[0,:,:,i])
        plt.axis('off')        
    pdf.savefig(fig)
    pdf.close()

    tf.reset_default_graph()


if __name__ == "__main__":
    main()
