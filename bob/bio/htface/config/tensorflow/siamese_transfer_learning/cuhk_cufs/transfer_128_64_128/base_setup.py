#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.bio.htface.config.tensorflow.utils import transfer_128_64_128 as transfer_architecture

# Updating the extracheckpoint to be non-trainable in InceptionV2
extra_checkpoint = {"checkpoint_path": "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/", 
                    "scopes": dict({"InceptionResnetV2/": "InceptionResnetV2/"}),
                    "trainable_variables": []
                   }


