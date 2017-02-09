#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.learn.linear
import bob.io.base
import numpy
import scipy.spatial

import logging
logger = logging.getLogger("bob.bio.htface")

from .HTAlgorithm import HTAlgorithm
from bob.learn.linear import GFKMachine, GFKTrainer
from .GFK import GFK
from bob.bio.face.algorithm import GaborJet
import math

class GFK_GaborJet(GFK, GaborJet):
  """
  
  Gabor jets + GFK

  
  """

  def __init__(
      self,
      number_of_subspaces,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      source_subspace_dimension,
      target_subspace_dimension,      
      use_lda=False, # BUild the subspaces with LDA
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      use_pinv=True,
      **kwargs  # parameters directly sent to the base class
  ):

      # For the time being, default parameters are fine
      GaborJet.__init__(self,
          # parameters for the tool
          'PhaseDiffPlusCanberra',
          multiple_feature_scoring = 'average',
          # some similarity functions might need a GaborWaveletTransform class, so we have to provide the parameters here as well...
          gabor_directions = 8,
          gabor_scales = 5,
          gabor_sigma = 2. * math.pi,
          gabor_maximum_frequency = math.pi / 2.,
          gabor_frequency_step = math.sqrt(.5),
          gabor_power_of_k = 0,
          gabor_dc_free = True
      )
      
      GFK.__init__(self,
         number_of_subspaces=number_of_subspaces,  # if int, number of subspace dimensions; if float, percentage of variance to keep
         source_subspace_dimension=source_subspace_dimension,
         target_subspace_dimension=target_subspace_dimension,
         use_lda=use_lda, # BUild the subspaces with LDA
         distance_function = distance_function,
         is_distance_function = is_distance_function,
         uses_variances = uses_variances,
         use_pinv=use_pinv,
         **kwargs
      )
      

      split_training_features_by_client = False
      if use_lda:
          split_training_features_by_client = True
      

  def _stack_gabor_jets(self, jets):
    """
    Stacking the absolute values of the gabor jets per node
    """

    jets_abs = {}
    jet_index = 0
    for j in jets:
        shape = (0, len(j[0].abs))
        jet_index = 0
        for a in j:
            if not jets_abs.has_key(jet_index):
                jets_abs[jet_index] = numpy.zeros(shape=shape)
            jets_abs[jet_index] = numpy.vstack((jets_abs[jet_index], a.abs))
            jet_index += 1
        
    return jets_abs


  def train_projector(self, training_features, projector_file):
    """Compute the kernel using the jets.abs"""


    gfk_trainer = GFKTrainer(self.m_number_of_subspaces, 
                             subspace_dim_source=self.m_source_subspace_dimension,
                             subspace_dim_target=self.m_target_subspace_dimension,
                             eps=self.eps)

    
    source_jets = training_features[0]
    target_jets = training_features[1]
    
    # Stacking the jets per node
    source_jets_abs = self._stack_gabor_jets(source_jets)
    target_jets_abs = self._stack_gabor_jets(target_jets)

    # Computing a kernel per node
    hdf5 = bob.io.base.HDF5File(projector_file, 'w')
    gfk_machine = []
    for k in source_jets_abs.keys():
        node_name = "node{0}".format(k)
        logger.info("  -> Training {0}".format(node_name))
        machine = gfk_trainer.train(source_jets_abs[k], target_jets_abs[k])
        hdf5.create_group(node_name)
        hdf5.cd(node_name)
        machine.save(hdf5)
        gfk_machine.append(machine)
        hdf5.cd("..")
    hdf5.set("nodes", len(source_jets_abs.keys()))

  def load_projector(self, projector_file):
    """Reads the1 PCA projection matrix from file"""
    # read PCA projector
    hdf5 = bob.io.base.HDF5File(projector_file, 'r')
    nodes = hdf5.get("nodes")
    self.gfk_machine = []
    hdf5 = bob.io.base.HDF5File(projector_file)    
    for k in range(nodes):
        node_name = "node{0}".format(k)
        hdf5.cd(node_name)
        self.gfk_machine.append(GFKMachine(hdf5))
        hdf5.cd("..")

  def project(self, feature):
    """Projects the data using the stored covariance matrix"""
    raise NotImplemented("There is no projection")

  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    Enrolls the model using one of several strategies.
    Commonly, the bunch graph strategy [WFK97]_ is applied, by storing several Gabor jets for each node.

    When ``multiple_feature_scoring = 'average_model'``, for each node the average :py:class:`bob.ip.gabor.Jet` is computed.
    Otherwise, all enrollment jets are stored, grouped by node.

    **Parameters:**

    enroll_features : [[:py:class:`bob.ip.gabor.Jet`]]
      The list of enrollment features.
      Each sub-list contains a full graph.

    **Returns:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The enrolled model.
      Each sub-list contains a list of jets, which correspond to the same node.
      When ``multiple_feature_scoring = 'average_model'`` each sub-list contains a single :py:class:`bob.ip.gabor.Jet`.
    """

    return GaborJet.enroll(self, enroll_features)

  def write_model(self, model, model_file):
    """Writes the model enrolled by the :py:meth:`enroll` function to the given file.

    **Parameters:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The enrolled model.

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for writing.
    """
    GaborJet.write_model(self, model, model_file)

  def read_model(self, model_file):
    """read_model(model_file) -> model

    Reads the model written by the :py:meth:`write_model` function from the given file.

    **Parameters:**

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for reading.

    **Returns:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The list of Gabor jets read from file.
    """
    return GaborJet.read_model(self, model_file)


  def read_probe(self, probe_file):
    """read_probe(probe_file) -> probe

    Reads the probe file, e.g., as written by the :py:meth:`bob.bio.face.extractor.GridGraph.write_feature` function from the given file.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for reading.

    **Returns:**

    probe : [:py:class:`bob.ip.gabor.Jet`]
      The list of Gabor jets read from file.
    """
    return GaborJet.read_probe(self, probe_file)


  def score(self, model, probe):
    """
    Compute the Kernalized scalar product between the absolute values of the Jets
    """

    local_scores = [numpy.dot(
                    numpy.dot(
                    (m[0].abs - machine.source_machine.input_subtract) / machine.source_machine.input_divide, machine.G), 
                    (p.abs - machine.target_machine.input_subtract) / machine.target_machine.input_divide)
                    for m, p ,machine in zip(model, probe, self.gfk_machine)]
    return numpy.average(local_scores)

