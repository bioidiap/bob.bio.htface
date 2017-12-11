.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

.. _bob.learn.tensorflow:

===============================
 Heterogeneous Face Recognition 
===============================

The goal of this package is to provide an "easy to reproduce" set of experiments in HETEROGENEOUS
face recognition databases.
This package is an extension of the
`bob.bio.base <https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/stable/index.html>`_ framework.

=============
 Installation
=============

The installation instructions are based on conda (**LINUX ONLY**).
Please `install conda <https://conda.io/docs/install/quick.html#linux-miniconda-install>`_ before continuing.

After everything installed do::

  $ cd bob.bio.htface
  $ conda env create -f environment.yml
  $ source activate bob.bio.htface  # activate the environment
  $ buildout


Before the magic begins, it's necessary to set a set of paths.
Please, edit this file according to your own working environment.
I hope the variable names are clear enough::

  $ vim ./bob/bio/htface/configs/base_paths.py

Follow below how this file looks like.

.. literalinclude:: ../bob/bio/htface/configs/base_paths.py
   :language: python
   :caption: "base_paths.py"

==========
The tasks
==========

.. Todo:: Describe the task


==========
Databases
==========

This subsection describes the databases used in this work.

.. toctree::
   :maxdepth: 2

   databases



==========
Hypotheses
==========

.. toctree::
   :maxdepth: 2

   session_variability
   gfk
   transfer_learning/transfer_learning


==========
User guide
==========

.. toctree::
   :maxdepth: 2

   user_guide


==========
References
==========
.. toctree::
   :maxdepth: 3

   references
