#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Plot the sum of the fft absolute values for each convolved signal

Usage:
  bob_htface_convolve_and_view.py  <output-file-name> <database> [--end-points=<arg>] [--demo]
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
from docopt import docopt
#from bob.learn.tensorflow.network import inception_resnet_v2, inception_resnet_v2_batch_norm
from bob.bio.htface.architectures.inception_v2_batch_norm import inception_resnet_v2_adapt_first_head

import tensorflow as tf
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from .registered_baselines import all_baselines, resources
import pkg_resources
from bob.bio.base.utils import read_config_file
from bob.bio.htface.utils import get_cnn_model_name
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def do_fft(image):
    img_complex = image.astype("complex128")
    img_fft = bob.sp.fft(img_complex)
    img_fft = abs(bob.sp.fftshift(img_fft))
    #psd = 10*numpy.log(abs(img_fft)**2)    
    return img_fft



def compute_histograms(convolved_image):
    """
    Given a set of convolved signals, stack the sum of the signal (of one conv) and return a vector
    
    Parameters
    ----------
      convolved_image: numpy.array
        Set of convolved images [n, w, h, c]
    
    """
    #import ipdb; ipdb.set_trace()
    #numpy.array([numpy.sum(do_fft(convolved_image[0,:,:,i])) for i in range(convolved_image.shape[3])])    
    return numpy.array([numpy.sum(do_fft(convolved_image[0,:,:,i])) for i in range(convolved_image.shape[3])])


def plot_images(raw_image, convolved_image, n_columns = 5):
    """
    PLot convolved images in the matplotlib
    """
    def normalize4save(img):
        norm_factor = numpy.max(img) - numpy.min(img)
        return (255 * ((img - numpy.min(img)) / (norm_factor))).astype("uint8")

    #n_rows = int(numpy.ceil(len(convolved_image)/float(n_columns)) + 1)
    n_rows = int(numpy.ceil(convolved_image.shape[3]/float(n_columns)) + 1)

    # Normalized convolved image 
    #norm_factor = numpy.sum((raw_image - numpy.mean(raw_image)) / numpy.std(raw_image))
    #convolved_image = convolved_image/norm_factor

    fig = plt.figure()
    plt.subplot(n_rows, n_columns, 1)
    plt.imshow(raw_image[0, :, :, 0], cmap='gray')
    plt.axis('off')    
    for i in range(convolved_image.shape[3]):
        plt.subplot(n_rows, n_columns, i+2)

        #if numpy.sum(convolved_image[0,:,:,i]) < 1e-10:
        #    for_printing = convolved_image[0,:,:,i]
        #else:
        #    for_printing = normalize4save(do_fft(convolved_image[0,:,:,i]))
        for_printing = normalize4save(do_fft(convolved_image[0,:,:,i]))
        #for_printing = normalize4save(convolved_image[0,:,:,i])
        
        plt.imshow(for_printing)
        plt.axis('off')
        
    return fig


def run_demo_mode(baselines, layers, database_name, config_base_path, output_path):
    """
    Plot the subfigures for the demo mode
    """

    ####### TODO: ADD AS A PARAMETER
    architecture = inception_resnet_v2_adapt_first_head

    logger.info("Demo mode !!!")
    
    baseline = baselines[0]
    #layer = layers[0]
    #layers = ["InceptionResnetV2/Conv2d_1a_3x3_left/",
    #          "InceptionResnetV2/Conv2d_1a_3x3_right/"]

    layers = ["Conv2d_1a_3x3_left",
              "Conv2d_1a_3x3_right"]
    

    # Loading the configuration setup
    config_preprocessing = os.path.join(resources[baseline]["preprocessed_data"], database_name+".py")
    config_database = resources["databases"][database_name]["config"]
    config = read_config_file([config_base_path, config_preprocessing, config_database])
    database = config.database

    for baseline in baselines:
    
        logger.info("Processing baseline {0} !!!".format(baseline))
        
        # Fetching model
        model_dir = get_cnn_model_name(config.temp_dir, resources[baseline]["name"],
                                       database_name, resources["databases"][database_name]["protocols"][0])

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
                  
        #scopes = {"InceptionResnetV2/Conv2d_1a_3x3_left/weights": "InceptionResnetV2/Conv2d_1a_3x3_left/weights",
        #          "InceptionResnetV2/Conv2d_1a_3x3_right/weights": "InceptionResnetV2/Conv2d_1a_3x3_right/weights"
        #          }                  
        
        tf.contrib.framework.init_from_checkpoint(model_dir, scopes)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        
        print(tf.global_variables()[0].eval(session)[0])
        
        
        #saver = tf.train.Saver()
        #if baseline == "idiap_casia_inception_v2_gray":
        #    model_dir = config.inception_resnet_v2_casia_webface_gray
        #saver.restore(session, tf.train.latest_checkpoint(model_dir))


        # getting the first sample only for the plot
        model_ids = None

        def normalize4save(img):
          return (255 * ((img - numpy.min(img)) / (numpy.max(img)-numpy.min(img)))).astype("uint8")
        
        #import ipdb; ipdb.set_trace();
        for modality, layer, end_points in zip(database.modalities, layers, [left_end_points, right_end_points]):
        
            file_object = database.objects(protocol=resources["databases"][database_name]["protocols"][0], groups="world", modality=[modality], model_ids=model_ids, purposes="train")[0]
            model_ids = [file_object.client_id]
            path = file_object.make_path(database.original_directory, ".hdf5")
            raw_image =  numpy.reshape(bob.io.base.load(path).astype("float32"), (1, 160, 160, 1))
            
            
            convolved_images = session.run(left_end_points["Conv2d_1a_3x3_left"], feed_dict={inputs: raw_image})
            fig = plot_images(raw_image, convolved_images)
            fig.savefig("XUXA_left_{0}.png".format(modality))

            convolved_images = session.run(right_end_points["Conv2d_1a_3x3_right"], feed_dict={inputs: raw_image})
            fig = plot_images(raw_image, convolved_images)
            fig.savefig("XUXA_right_{0}.png".format(modality))






            #raw_image =  numpy.reshape(bob.io.base.load(path).astype("float32"), (1, 160, 160, 1))

           # convolved_images, convolved_images_bias = dump_and_convolve(raw_image, session, layer)

            # Convolving the first layer
            #import ipdb; ipdb.set_trace();
            

            x=0
            #fig.savefig(os.path.join(output_path,"{0}_{1}_{2}.png").format(database_name, modality, baseline))
        
            #fig = plot_images(raw_image, convolved_images_bias)
            #fig.savefig(os.path.join(output_path,"{0}_{1}_{2}_bias.png").format(database_name, modality, baseline))


        tf.reset_default_graph()


def main():

    args = docopt(__doc__, version='Run experiment')
    set_verbosity_level(logger, 2)
    output_file_name = args["<output-file-name>"]    
    database_name = args["<database>"]

    #baselines = ["idiap_casia_inception_v2_gray",
    #             "idiap_casia_inception_v2_gray_adapt_first_layer",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_2",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_4",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_5",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_6",
    #             "idiap_casia_inception_v2_gray_adapt_all_layers"]


    #baselines = ["idiap_casia_inception_v2_gray",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_6"]

    #baselines = ["idiap_casia_inception_v2_gray",
    #             "idiap_casia_inception_v2_gray_adapt_layers_1_6"]

    baselines = ["idiap_casia_inception_v2_gray_adapt_first_layer_nonshared_batch_norm"]

    # Loading base paths
    config_base_path = pkg_resources.resource_filename("bob.bio.htface",
                                                       "configs/base_paths.py")
    base_paths = read_config_file([config_base_path])

    pdf = PdfPages(output_file_name)
    #layers = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"]
    #layers = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",  "Mixed_5b", "Block35"]
    #layers = ["Conv2d_1a_3x3", "Conv2d_2a_3x3"]
    layers = ["Conv2d_1a_3x3"]
    
    # Checking if demo mode
    if args["--demo"]:
        run_demo_mode(baselines, layers, database_name, config_base_path, os.path.dirname(output_file_name))
        return


    for layer in layers:
        logger.info("Processing layer {0}".format(layer))

        # Bootstraping the plot
        fig = plt.figure()        
        fig.suptitle("Layer {0}:".format(layer))

        for baseline in baselines:
            logger.info("Processing baseline {0}".format(baseline))

            # Fetching model
            model_dir = get_cnn_model_name(base_paths.temp_dir, resources[baseline]["name"],
                                           database_name, resources["databases"][database_name]["protocols"][0])


            # Loading database object
            config_preprocessing = os.path.join(resources[baseline]["preprocessed_data"], database_name+".py")
            config_database = resources["databases"][database_name]["config"]
            config = read_config_file([config_base_path, config_preprocessing, config_database])
            database = config.database

            # Loading TF MODEL
            inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))

            # Getting the end_points
            prelogits, end_points = inception_resnet_v2(tf.stack([tf.image.per_image_standardization(i) for i in tf.unstack(inputs)]),
                                              mode=tf.estimator.ModeKeys.PREDICT)
            
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            if baseline == "idiap_casia_inception_v2_gray":
                model_dir = config.inception_resnet_v2_casia_webface_gray

            saver.restore(session, tf.train.latest_checkpoint(model_dir))
            
            hist_modality_A = None
            hist_modality_B = None        
            counter_A = 0
            counter_B = 0
            
            for o in database.objects(protocol=resources["databases"][database_name]["protocols"][0], groups="world"):
                path = o.make_path(database.original_directory, ".hdf5")
                image = bob.io.base.load(path).astype("float32")
                image = numpy.reshape(image, (1, image.shape[0], image.shape[1], 1))
                
                data = session.run(end_points[layer], feed_dict={inputs: image})
                hist = compute_histograms(data)
                
                if o.modality == database.modalities[0]:
                    if hist_modality_A is None:
                        hist_modality_A = hist
                    else:
                        hist_modality_A = hist_modality_A +  hist
                    counter_A += 1
                        
                else:
                    if hist_modality_B is None:
                        hist_modality_B = hist
                    else:
                        hist_modality_B = hist_modality_B + hist
                    counter_B += 1

            max_lim = max(numpy.max(hist_modality_A/counter_A), numpy.max(hist_modality_B/counter_B))
            
            plt.plot(hist_modality_A/counter_A, '^', linewidth=0.0, label=database.modalities[0] +"_" + baseline, alpha=0.30)
            plt.plot(hist_modality_B/counter_B, 's', linewidth=0.0, label=database.modalities[1] +"_" + baseline, alpha=0.30)

            #import ipdb; ipdb.set_trace()
            for i in range(hist_modality_A.shape[0]):
                plt.plot([i, i], [hist_modality_A[i]/counter_A, hist_modality_B[i]/counter_B], 'k-', lw=0.5, alpha=0.5)
            
            tf.reset_default_graph()

        plt.grid(True)  
        plt.legend(prop={'size': 6})
        pdf.savefig(fig)
        
       
    pdf.close()


if __name__ == "__main__":
    main()
