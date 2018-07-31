#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.bio.base.baseline import Baseline
import pkg_resources

# IMPORTING THE NECESSARY MODULES

# Siamese
import bob.bio.htface.configs.style_domain_specific_units.siamese_transfer_learning.inception_resnet_v2_adapt_layers_1_6.estimator_nonshared_batch_norm


class SiameseAdaptLayers1_6_BatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the 1-6 layers
    """

    def __init__(self, **kwargs):
    
        name              = "styledsu_siamese_inceptionv2_adapt_1_6_nonshared_batch_norm"
        extractor       = pkg_resources.resource_filename("bob.bio.htface", "configs/style_domain_specific_units/siamese_transfer_learning/inception_resnet_v2_adapt_layers_1_6/extractor_nonshared_batch_norm.py")
        
        preprocessors   = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/inception_resnet_v2_gray_preprocessor.py")}
        algorithm         = "distance-cosine"

        self.baseline_type     = "Siamese BN"
        self.reuse_extractor   = False        

        # train cnn
        self.estimator         = bob.bio.htface.configs.style_domain_specific_units.siamese_transfer_learning.inception_resnet_v2_adapt_layers_1_6.estimator_nonshared_batch_norm.get_estimator
        self.preprocessed_data = pkg_resources.resource_filename("bob.bio.htface", "configs/tensorflow/siamese_transfer_learning/inception_resnet_v2_databases/")

        super(SiameseAdaptLayers1_6_BatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)



# Entry points
styledsu_siamese_inceptionv2_adapt_1_6_nonshared_batch_norm = SiameseAdaptLayers1_6_BatchNorm()

