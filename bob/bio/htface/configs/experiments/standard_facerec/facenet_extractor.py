#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.ip.tensorflow_extractor import FaceNet
from bob.bio.base.extractor import CallableExtractor
    
#########
# Extraction
#########

extractor = CallableExtractor(FaceNet())
