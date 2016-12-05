#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.learn.linear
import bob.io.base
import numpy
import scipy.spatial
from bob.bio.base.algorithm import Algorithm


class HT_LDA (Algorithm):
  """Tool for computing eigenfaces"""

  def __init__(
      self,
      subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      use_pinv=True,
      **kwargs  # parameters directly sent to the base class
  ):

    # call base class constructor and register that the algorithm performs a projection
    Algorithm.__init__(
        self,
        performs_projection = True,

        subspace_dimension = subspace_dimension,
        distance_function = distance_function,
        is_distance_function = is_distance_function,
        uses_variances = uses_variances,

        **kwargs
    )

    self.m_subspace_dim = subspace_dimension
    self.m_machine = None
    self.m_distance_function = distance_function
    self.m_factor = -1. if is_distance_function else 1.
    self.m_uses_variances = uses_variances
    self.m_use_pinv = use_pinv


  def __train_pca(self, data, max_energy=0.99):
    """
    Trains the PCA and returns the eigenvector matrix with ``max_energy'' kept
    """

    t = bob.learn.linear.PCATrainer()
    machine, variances = t.train(data)

    # compute variance percentage, if desired
    cummulated = numpy.cumsum(variances) / numpy.sum(variances)
    dim_kept = 0
    for index in range(len(cummulated)):
      if cummulated[index] > max_energy:
        dim_kept = index+1
        break

    machine.resize(machine.shape[0], dim_kept)
    return machine


  def __train_lda(self, data, max_energy=0.99):
    """
    Trains the PCA and returns the eigenvector matrix with ``max_energy'' kept
    """

    t = bob.learn.linear.FisherLDATrainer(use_pinv=self.m_use_pinv)
    machine, variances = t.train(data)

    # compute variance percentage, if desired
    cummulated = numpy.cumsum(variances) / numpy.sum(variances)
    dim_kept = 0
    for index in range(len(cummulated)):
      if cummulated[index] > max_energy:
        dim_kept = index+1
        break

    machine.resize(machine.shape[0], dim_kept)
    return machine


  def train_projector(self, training_features, projector_file):
    """Generates the PCA covariance matrix"""
           
    # Initializes the data
    data_A = numpy.vstack([feature.flatten() for feature in training_features[0]])
    data_B = numpy.vstack([feature.flatten() for feature in training_features[1]])
  
  
    utils.info("  -> Training LinearMachine using PCA")
    
    pca_machine = self.__train_pca(numpy.concatenate((data_A,data_B),axis=0))
    
    #Projecting the input data in the PCA space
    data_A = pca_machine.forward(data_A)
    data_B = pca_machine.forward(data_B)

    utils.info("  -> Training LinearMachine using LDA")

    lda_machine = self.__train_lda([data_A,data_B])

    #Combining the PCA and the LDA machine
    self.m_machine                = bob.learn.linear.Machine(input_size=pca_machine.shape[0], output_size=lda_machine.shape[1])
    self.m_machine.input_subtract = pca_machine.input_subtract
    self.m_machine.weights        = numpy.dot(pca_machine.weights, lda_machine.weights)
    
    utils.info("    ... Keeping %d LDA dimensions" % self.m_subspace_dim)

    f = bob.io.base.HDF5File(projector_file, "w")
    self.m_machine.save(f)
    del f


  def load_projector(self, projector_file):
    """Reads the PCA projection matrix from file"""
    # read PCA projector
    f = bob.io.base.HDF5File(projector_file)
    self.m_machine = bob.learn.linear.Machine(f)
    del f
    # Allocates an array for the projected data

  def project(self, feature):
    """Projects the data using the stored covariance matrix"""
    return self.m_machine(feature)

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    assert len(enroll_features)
    # just store all the features
    model = numpy.zeros((len(enroll_features), enroll_features[0].shape[0]), numpy.float64)
    for n, feature in enumerate(enroll_features):
      model[n,:] += feature[:]

    # return enrolled model
    return model


  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function taken from the config file"""
    # return the negative distance (as a similarity measure)
    if len(model.shape) == 2:
      # we have multiple models, so we use the multiple model scoring
      return self.score_for_multiple_models(model, probe)
    elif self.m_uses_variances:
      # single model, single probe (multiple probes have already been handled)
      return self.m_factor * self.m_distance_function(model, probe, self.m_variances)
    else:
      # single model, single probe (multiple probes have already been handled)
      return self.m_factor * self.m_distance_function(model, probe)
