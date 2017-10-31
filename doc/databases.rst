.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


=============================
 Heterogeneous Face Databases
=============================


CUHK Face Sketch Database (CUFS)
--------------------------------


CUHK Face Sketch database (`CUFS <http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html>`_) is composed by viewed sketches.
It includes 188 faces from the Chinese University of Hong Kong (CUHK) student database, 123 faces from the `AR database <http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html>`_ and 295 faces from the `XM2VTS database <http://www.ee.surrey.ac.uk/CVSSP/xm2vtsdb/>`_.

There are 606 face images in total. 
For each face image, there is a sketch drawn by an artist based on a photo taken in a frontal pose, under normal lighting condition and with a neutral expression.

There is no evaluation protocol established for this database.
Each work that uses this database implements a different way to report the results.
In [Wang2009]_ the 606 identities were split in three sets (153 identities for training, 153 for development, 300 for evaluation).
The rank-1 identification rate in the evaluation set is used as performance measure.
Unfortunately the file names for each set were not distributed.

In [Klare2013]_ the authors created a protocol based on a 5-fold cross validation splitting the 606 identities in two sets with 404 identities for training and 202 for testing.
The average rank-1 identification rate is used as performance measure.
In [Bhatt2012]_, the authors evaluated the error rates using only the pairs (VIS -- Sketch) corresponding to the CUHK Student Database and AR Face Database and in [Bhatt2010]_ the authors used only the pairs corresponding to the CUHK Student Database.
In [Yi2015]_ the authors created a protocol based on a 10-fold cross validation splitting the 606 identities in two sets with 306 identities for training and 300 for testing.
Also the average rank-1 identification error rate in the test is used to report the results.
Finally in [Roy2016]_, since the method does not requires a background model, the whole 606 identities were used for evaluation and also to tune the hype-parameters; which is not a good practice in machine learning.
Just by reading what is written in the paper (no source code available), we can claim that the evaluation is biased.

For comparison reasons, we will follow the same strategy as in [Klare2013]_ and do a 5 fold cross-validation splitting the 606 identities in two sets with 404 identities for training and 202 for testing and use the average rank-1 identification rate, in the evaluation set as a metric.
For reproducibility purposes, this evaluation protocol is published in a python package `format <https://pypi.python.org/pypi/bob.db.cuhk_cufs>`_`.
In this way future researchers will be able to reproduce exactly the same tests with the same identities in each fold (which is not possible today).


CASIA NIR-VIS 2.0 face database
-------------------------------

CASIA NIR-VIS 2.0 database [Li2013]_ offers pairs of mugshot images and their correspondent NIR photos. 
The images of this database were collected in four recording sessions: 2007 spring, 2009 summer, 2009 fall and 2010 summer, in which the first session is identical to the CASIA HFB database [Li2009]_. 
It consists of 725 subjects in total. 
There are [1-22] VIS and [5-50] NIR face images per subject.
The eyes positions are also distributed with the images.

This database has a well defined protocol and it is publicly available for `download <http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html>`_.
We also organized this protocol in the same way as for CUFS database and it is also freely available for download `(bob.db.cbsr_nir_vis_2) <https://pypi.python.org/pypi/bob.db.cbsr_nir_vis_2>`_.
The average rank-1 identification rate in the evaluation set (called view 2) is used as an evaluation metric.



CUHK Face Sketch FERET Database (CUFSF)
---------------------------------------

The CUHK Face Sketch FERET Database (CUFSF) is composed by viewed sketches.
It includes 1,194 face images from the `FERET database <http://www.itl.nist.gov/iad/humanid/feret/>`_ and theirs respectively sketch draw by an artist.

There is not an evaluation protocol established for this database.
Each work that uses this database implements a different way to report the results.
In [Zhang2011]_ the authors split the 1,194 identities in two sets with 500 identities for training and 694 for testing.
Unfortunately the file names for each set was not distributed.
The Verification Rate (**VR**) considering a False Acceptance Rate ($FAR$) of 0.1\% is used as a performance measure.
In [Lei2012]_ the authors split the 1,194 identities in two sets with 700 identities for training and 494 for testing.
The rank-1 identification rate is used as performance measures.


Long Distance Heterogeneous Face Database
-----------------------------------------

Long Distance Heterogeneous Face Database (LDHF-DB) contains pairs of VIS and NIR face images at distances of 60m, 100m, and 150m outdoors and at a 1m distance indoors of 100 subjects (70 males and 30 females).
For each subject one image was captured at each distance in daytime and nighttime. 
All the images of individual subjects are frontal faces without glasses, and collected in a single sitting.

The short distance visible light images (1m) were collected under a fluorescent light by using a DSLR camera with Canon F1.8 lens, and NIR images were collected using the modified DSLR camera and NIR illuminator of 24 IR LEDs without visible light.
Long distance (over 60m) VIS images were collected during the daytime using a telephoto lens coupled with a DSLR camera, and NIR images were collected using the DSLR camera with NIR light provided by RayMax300 illuminator.

For evaluation purposes, the authors of the database [Kang2014]_ defined a 10-fold cross validation with 90 subjects for training and 10 subjects for testing.
ROC (Receiver Operating Characteristic) and CMC (Cumulative Match Characteristic) were used for comparison.

Pola Thermal
------------

Describe ...






