#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.ip.gabor
import bob.ip.base
import xfacereclib.extension.HTUBM

import numpy
import math

from facereclib.features.Extractor import Extractor
from facereclib import utils


class LBPHS (Extractor):
  """Extractor for local binary pattern histogram sequences"""

  def __init__(
      self,
      # Block setup
      block_size,    # one or two parameters for block size
      block_overlap = 0, # one or two parameters for block overlap
      # LBP parameters
      lbp_radius = 2,
      lbp_neighbor_count = 8,
      lbp_uniform = True,
      lbp_circular = True,
      lbp_rotation_invariant = False,
      lbp_compare_to_average = False,
      lbp_add_average = False,
      # histogram options
      sparse_histogram = False,
      split_histogram = None
  ):
    """Initializes the local binary pattern histogram sequence tool chain with the given file selector object"""

    # call base class constructor
    Extractor.__init__(
        self,

        block_size = block_size,
        block_overlap = block_overlap,
        lbp_radius = lbp_radius,
        lbp_neighbor_count = lbp_neighbor_count,
        lbp_uniform = lbp_uniform,
        lbp_circular = lbp_circular,
        lbp_rotation_invariant = lbp_rotation_invariant,
        lbp_compare_to_average = lbp_compare_to_average,
        lbp_add_average = lbp_add_average,
        sparse_histogram = sparse_histogram,
        split_histogram = split_histogram
    )

    self.m_lbp = bob.ip.base.LBP (lbp_neighbor_count, radius=lbp_radius, to_average=lbp_add_average, uniform=lbp_uniform, rotation_invariant=lbp_rotation_invariant)
    self.block_size=block_size
    self.block_overlap = block_overlap
    

    #self.m_lbp = bob.ip.base.LBP (lbp_neighbor_count, block_size=block_size, block_overlap=block_overlap, to_average=lbp_add_average, uniform=lbp_uniform, rotation_invariant=lbp_rotation_invariant)
    


  def __call__(self, image):
    """Extracts the local Gabor binary pattern histogram sequence from the given image"""

    #lbp_image = self.m_lbp(image)
    #import ipdb; ipdb.set_trace();
    
    hist = bob.ip.base.lbphs(image, self.m_lbp, block_size=self.block_size, block_overlap=self.block_overlap).flatten().astype('float')
    #hist      = bob.ip.base.histogram(lbp_image, self.m_lbp.max_label)
    #hist      = numpy.array(hist, dtype=numpy.float64)
    #hist      /= sum(hist) #normalizing
    
    # return the concatenated list of all histograms
    #return self.m_lbp(image)
    return hist

