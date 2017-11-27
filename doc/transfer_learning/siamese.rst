.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>



Preamble
--------

From the experiments in :ref:`first insights section <first-insights>`_ it was possible to observe that a CNN model
trained with only visible images provided recognition rates far from being random, but still with very low
if compared with the state-of-the-art.

This section we will explore strategies on how to use such prior and adapt our :math:`\phi` to some target image modality.
More preciselly, we'll use Siamese networks [Chopra2005]_. **VERY BAD TEXT**

Siamese Neural Networks (**SNN**) learn the non-linear subspaces :math:`\phi` by repeatedly presenting
pairs of positive and negative examples (belonging to the same class or not).
To our particular task, the pairs of examples belong to clients sensed by different image modalities
(:math:`X_A` and :math:`X_B` ) as we can observe in the figure below.

.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %
,  
where :math:`L` is defined as :math:`L (X_A, X_B) = || \phi_{\theta_1}(X_A) - \phi_{\theta_1}(X_B)||`.


.. is as small as possible when $X_A$ and $X_B$ belong to the same client and as large as possible otherwise.
.. In this case $\phi$ is the output of the CNN.
.. It is important to highlight that the weights $W$ are the same for both inputs (they share the same subspace) and this is the reason why the architecture is called Siamese.





.. Red trainable layers
.. Blue non trainable layers



Classic transfer
-----------------

In this stage we will apply the classic way

.. image:: ../plots/transfer-learning/siamese.png
  :scale: 100 %




Domain specific embedding
-------------------------


Adapting layer by layer
-----------------------

