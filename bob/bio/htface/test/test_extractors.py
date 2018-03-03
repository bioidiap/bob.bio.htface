#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import numpy
from bob.bio.htface.extractor import MLBPHS
numpy.random.seed(10)


def test_MLBPHS():

    fake_image = (numpy.random.rand(160, 160)*255).astype("uint8")
    
    lbphs = MLBPHS(block_size=(64, 64),
                  lbp_radius=[1, 2, 4],
                  lbp_uniform=True,
                  lbp_circular=True
                  )

    lbphs = lbphs(fake_image)
    assert lbphs.shape[0] == 708

