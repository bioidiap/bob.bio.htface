#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pkg_resources

all_baselines = ["idiap_casia_inception_v2_gray",
                 "idiap_casia_inception_v2_gray_transfer_64_128",
                 "idiap_casia_inception_v2_gray_adapt_first_layer",
                 "idiap_casia_inception_v2_gray_adapt_layers_1_2",
                 "idiap_casia_inception_v2_gray_adapt_layers_1_4",
                 "idiap_casia_inception_v2_gray_adapt_layers_1_5",
                 "idiap_casia_inception_v2_gray_adapt_layers_1_6",
                 "idiap_casia_inception_v2_gray_adapt_all_layers"]

resources = dict()

# Mapping databases
resources["databases"] = dict()
resources["databases"]["cuhk_cufs"] = dict()
resources["databases"]["cuhk_cufs"]["config"] = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/cuhk_cufs.py")
resources["databases"]["cuhk_cufs"]["protocols"] = ["search_split1_p2s", "search_split2_p2s", "search_split3_p2s", "search_split4_p2s", "search_split5_p2s"]
resources["databases"]["cuhk_cufs"]["groups"] = ["dev"]

resources["databases"]["cuhk_cufsf"] = dict()
resources["databases"]["cuhk_cufsf"]["config"] = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/cuhk_cufsf.py")
resources["databases"]["cuhk_cufsf"]["protocols"] = ["search_split1_p2s", "search_split2_p2s", "search_split3_p2s", "search_split4_p2s", "search_split5_p2s"]
resources["databases"]["cuhk_cufsf"]["groups"] = ["dev"]

resources["databases"]["nivl"] = dict()
resources["databases"]["nivl"]["config"] = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/nivl.py")
resources["databases"]["nivl"]["protocols"] = ["idiap-search_VIS-NIR_split1", "idiap-search_VIS-NIR_split2", 
                                               "idiap-search_VIS-NIR_split3", "idiap-search_VIS-NIR_split4", "idiap-search_VIS-NIR_split5"]
resources["databases"]["nivl"]["groups"] = ["dev"]

resources["databases"]["casia_nir_vis"] = dict()
resources["databases"]["casia_nir_vis"]["config"] = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/casia_nir_vis.py")
resources["databases"]["casia_nir_vis"]["protocols"] = ["view2_1", "view2_2","view2_3","view2_4","view2_5",
                                                        "view2_6", "view2_7", "view2_8", "view2_9", "view2_10"]
resources["databases"]["casia_nir_vis"]["groups"] = ["eval"]

resources["databases"]["pola_thermal"] = dict()
resources["databases"]["pola_thermal"]["config"] = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/pola_thermal.py")
resources["databases"]["pola_thermal"]["protocols"] = ["VIS-polarimetric-overall-split1", "VIS-polarimetric-overall-split2", "VIS-polarimetric-overall-split3",
                                                       "VIS-polarimetric-overall-split4", "VIS-polarimetric-overall-split5"]
resources["databases"]["pola_thermal"]["groups"] = ["dev"]

# idiap_casia_inception_v2_gray
resources["idiap_casia_inception_v2_gray"] = dict()
resources["idiap_casia_inception_v2_gray"]["name"] = "idiap_casia_inception_v2_gray"
resources["idiap_casia_inception_v2_gray"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/extractor.py")
resources["idiap_casia_inception_v2_gray"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray"]["reuse_extractor"] = True

# INCEPTION_V2 + transfer 64-128
resources["idiap_casia_inception_v2_gray_transfer_64_128"] = dict()
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["name"] = "idiap_casia_inception_v2_gray_transfer_64_128"
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_transfer_64_128/extractor.py")
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_transfer_64_128/estimator.py")
resources["idiap_casia_inception_v2_gray_transfer_64_128"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")



# INCEPTION_V2 + first layer
resources["idiap_casia_inception_v2_gray_adapt_first_layer"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["name"] = "idiap_casia_inception_v2_gray_adapt_first_layer"
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_first_layer/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_first_layer/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_first_layer"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")



# INCEPTION_V2 + first and second layers
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["name"] = "idiap_casia_inception_v2_gray_adapt_layers_1_2"
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_layers_1_2/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_layers_1_2/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_2"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")



# INCEPTION_V2 + first and forth layers
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["name"] = "idiap_casia_inception_v2_gray_adapt_layers_1_4"
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_layers_1_4/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_layers_1_4/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_4"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")




# INCEPTION_V2 + first and fifth layers
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["name"] = "idiap_casia_inception_v2_gray_adapt_layers_1_5"
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_layers_1_5/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_layers_1_5/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_5"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")


# INCEPTION_V2 + first and sixth layers
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["name"] = "idiap_casia_inception_v2_gray_adapt_layers_1_6"
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_layers_1_6/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_layers_1_6/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_layers_1_6"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")





# INCEPTION_V2 + ALL LAYERS
resources["idiap_casia_inception_v2_gray_adapt_all_layers"] = dict()
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["name"] = "idiap_casia_inception_v2_gray_adapt_all_layers"
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["extractor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2_adapt_all_layers/extractor.py")
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["preprocessor"] = pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/inception_resnet_v2/preprocessor.py")
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["reuse_extractor"] = False

## To train the cnn
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["estimator"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_adapt_all_layers/estimator.py")
resources["idiap_casia_inception_v2_gray_adapt_all_layers"]["preprocessed_data"] = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")



