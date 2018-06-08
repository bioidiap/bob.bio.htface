#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.bio.base.baseline import Baseline
import pkg_resources
import bob.bio.htface

# IMPORTING THE NECESSARY MODULES

# Siamese
import bob.bio.htface.configs.free_domain_specific_units.siamese_transfer_learning.inception_resnet_v2_adapt_first_layer.estimator_nonshared_batch_norm


class SiameseAdaptFirstBatchNorm(Baseline):
    """
    This baseline has the following features:
      - The prior uses batch norm in all layers
      - Siamese net
      - Adapt the first layer only
    """

    def __init__(self, **kwargs):
    
        name            = "fdsu_siamese_inceptionv2_first_layer_nonshared_batch_norm"
        extractor       = pkg_resources.resource_filename("bob.bio.htface", "configs/free_domain_specific_units/siamese_transfer_learning/inception_resnet_v2_adapt_first_layer/extractor_nonshared_batch_norm.py")
        preprocessors   = {"default": pkg_resources.resource_filename("bob.bio.htface", "configs/experiments/standard_facerec/inception_resnet_v2_gray_preprocessor.py")}
        algorithm       = "distance-cosine"

        self.reuse_extractor   = False        
        self.baseline_type     = "Siamese BN"

        # train cnn
        self.estimator         = bob.bio.htface.configs.free_domain_specific_units.siamese_transfer_learning.inception_resnet_v2_adapt_first_layer.estimator_nonshared_batch_norm.get_estimator 
        self.preprocessed_data = None

        super(SiameseAdaptFirstBatchNorm, self).__init__(name, preprocessors, extractor, algorithm, **kwargs)

        
        
fdsu_inception_resnet_v2_siamese_adapt_first = SiameseAdaptFirstBatchNorm()

