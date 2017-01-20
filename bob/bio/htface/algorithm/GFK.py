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


class GFK (HTAlgorithm):
  """
  
  Implementing the algorithm Geodesic Flow Kernel to do transfer learning from the modality A to modality B from the paper
  
  Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.  
  
  
  A very good explanation can be found here
  
  http://www-scf.usc.edu/~boqinggo/domainadaptation.html#gfk_section
  
  """

  def __init__(
      self,
      subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      principal_angles_dimension,
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      use_pinv=True,
      **kwargs  # parameters directly sent to the base class
  ):

    # call base class constructor and register that the algorithm performs a projection
    HTAlgorithm.__init__(self,
        performs_projection = False, # enable if your tool will project the features
        requires_projector_training = True, # by default, the projector needs training, if projection is enabled
        split_training_features_by_client = False, # enable if your projector training needs the training files sorted by client
        split_training_features_by_modality = True, # enable if your projector training needs the training files sorted by modality      
        use_projected_features_for_enrollment = False, # by default, the enroller used projected features for enrollment, if projection is enabled.
        requires_enroller_training = False, # enable if your enroller needs training

        subspace_dimension = subspace_dimension,
        distance_function = distance_function,
        is_distance_function = is_distance_function,
        **kwargs
    )

    self.m_subspace_dim = subspace_dimension
    self.m_machine = None
    self.m_distance_function = distance_function
    self.m_factor = -1. if is_distance_function else 1.
    self.m_uses_variances = uses_variances
    self.m_principal_angles_dimension = principal_angles_dimension
    
    self.source_machine = None
    self.target_machine = None
    self.G = None
    self.requires_projector_training = True
    self.eps = 1e-20;


  def _null_space(self, A):
    """
      [~,S,V] = svd(A,0);
      if m > 1, s = diag(S);
         elseif m == 1, s = S(1);
         else s = 0;
      end
      tol = max(m,n) * eps(max(s));
      r = sum(s > tol);
      Z = V(:,r+1:n);    
    """

    U, S, V = numpy.linalg.svd(A)
    m, n = A.shape
    
    if m==1:
      s = S[0]
    else:
      s = S

    r = sum(s > self.eps)
    return V[:,r:n]


  def _train_gfk(self, Ps, Pt):
    """
    Trains the PCA and returns the eigenvector matrix with ``max_energy'' kept
    """

    import numpy

    N = Ps.shape[1]
    dim = Pt.shape[1]
    

    #Principal angles
    QPt = numpy.dot(Ps.T, Pt)
     
    #[V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    A = QPt[0:dim,:].copy()
    B = QPt[dim:,:].copy()
    
    [V1,V2,V,Gam,Sig] = bob.math.gsvd(A, B)
    V2 = -V2
    
    # Some sanity checks
    I = numpy.eye(V1.shape[1])
    I_check = numpy.dot(Gam.T, Gam) + numpy.dot(Sig.T, Sig)
    assert numpy.sum(abs(I-I_check)) < 1e-10

    
    theta = numpy.arccos(numpy.diagonal(Gam))
    
    B1 = numpy.diag(0.5* (1+( numpy.sin(2*theta) / (2.*numpy.maximum
  (theta,self.eps)))))
    B2 = numpy.diag(0.5*((numpy.cos(2*theta)-1) / (2*numpy.maximum(
  theta,self.eps))))
    B3 = B2
    B4 = numpy.diag(0.5* (1-( numpy.sin(2*theta) / (2.*numpy.maximum
  (theta,self.eps)))))


    delta1_1 = numpy.hstack( (V1, numpy.zeros(shape=(dim,N-dim))) )
    delta1_2 = numpy.hstack( (numpy.zeros(shape=(N-dim, dim)), V2) )
    delta1 = numpy.vstack((delta1_1, delta1_2))

    delta2_1 = numpy.hstack( (B1, B2,numpy.zeros(shape=(dim,N-2*dim)  )))
    delta2_2 = numpy.hstack( (B3, B4,numpy.zeros(shape=(dim,N-2*dim)  )))
    delta2_3 = numpy.zeros(shape=(N-2*dim, N))
    delta2 = numpy.vstack((delta2_1, delta2_2, delta2_3))

    delta3_1 = numpy.hstack((V1, numpy.zeros(shape=(dim,N-dim))))
    delta3_2 = numpy.hstack( (numpy.zeros(shape=(N-dim, dim)), V2))
    delta3 = numpy.vstack((delta3_1, delta3_2)).T

    delta = numpy.dot(numpy.dot(delta1, delta2), delta3)  
    G = numpy.dot(numpy.dot(Ps, delta), Ps.T)
      
    return G


  def _znorm(self, data):
    """
    Z-Normaliza
    """

    mu  = numpy.average(data,axis=0)
    std = numpy.std(data,axis=0)

    data = (data-mu)/std

    return data,mu,std

  def _train_pca(self, data, mu_data, std_data):
    t = bob.learn.linear.PCATrainer()
    machine, variances = t.train(data)

    # For re-shaping, we need to copy...
    variances = variances.copy()
    subspace_dim = self.m_subspace_dim
    
    # compute variance percentage, if desired
    if isinstance(self.m_subspace_dim, float):
      cummulated = numpy.cumsum(variances) / numpy.sum(variances)
      for index in range(len(cummulated)):
        if cummulated[index] > subspace_dim:
          subspace_dim = index
          break
      subspace_dim = index
    logger.info("    ... Keeping %d PCA dimensions", subspace_dim)
    
    machine.resize(machine.shape[0], subspace_dim)
    machine.input_subtract = mu_data
    machine.input_divide = std_data
    
    return machine


  def train_projector(self, training_features, projector_file):
    """Compute the kernel"""

    source = training_features[0]
    target = training_features[1]

    logger.info("  -> Normalizing data per modality")
    source, mu_source, std_source = self._znorm(source)
    target, mu_target, std_target = self._znorm(target)

    logger.info("  -> Computing PCA for the source modality")
    Ps = self._train_pca(source, mu_source, std_source)
    logger.info("  -> Computing PCA for the target modality")
    Pt = self._train_pca(target, mu_target, std_target)
    #self.m_machine                = bob.io.base.load("/idiap/user/tpereira/gitlab/workspace_HTFace/GFK.hdf5")
    
    G = self._train_gfk(numpy.hstack((Ps.weights, self._null_space(Ps.weights.T))), Pt.weights[:,0:self.m_principal_angles_dimension])
    
    # Saving the source linear machine, target linear machine and the Kernel
    f = bob.io.base.HDF5File(projector_file, "w")
    f.create_group("source_machine")
    f.cd("/source_machine")
    Ps.save(f)
    f.cd("..")
    f.create_group("target_machine")
    f.cd("/target_machine")
    Pt.save(f)
    f.cd("..")    
    f.set("G", G)
    
    self.source_machine = Ps
    self.target_machine = Pt
    self.G = G
    
    del f

  def load_projector(self, projector_file):
    """Reads the PCA projection matrix from file"""
    # read PCA projector
    f = bob.io.base.HDF5File(projector_file, 'r')
    f.cd("/source_machine")
    self.source_machine = bob.learn.linear.Machine(f)
    f.cd("..")
    f.cd("/target_machine")
    self.target_machine = bob.learn.linear.Machine(f)
    f.cd("..")
    self.G = f.get("G")
    del f
    
    # Allocates an array for the projected data

  def project(self, feature):
    """Projects the data using the stored covariance matrix"""
    raise NotImplemented("There is no projection")

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
    model = (model-self.source_machine.input_subtract) / self.source_machine.input_divide
    probe = (probe-self.target_machine.input_subtract) / self.target_machine.input_divide

    return numpy.dot(numpy.dot(model,self.G), probe.T)[0]

