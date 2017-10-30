#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@partner.samsung.com>

"""
Plot t-SNE clusted by modality

t-SNE is a tool to visualize high-dimensional data. 
It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler 
divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

WARNING: This is not a convex problem, so please try different seeds (`--seed` option).

More info in:
van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.


For information about <database> parameter plese do:
  resources.py --type database


Usage:
  plot_tsne.py  <database> <protocol> --database-original-directory=<arg>
               [--database-extension=<arg> --output-file=<arg>]
               [--iterations=<arg> --learning-rate=<arg> --perplexity=<arg> --seed=<arg>]
  plot_tsne.py -h | --help

Options:
  --output-file=<arg>                   Output file [default: tsne.pdf]
  --iterations=<arg>                    Maximum number of iterations for the gradient descend [default: 5000]
  --learning-rate=<arg>                 Learning rate for the gradient descend [default: 100.0]
  --perplexity=<arg>                    Perplexity [default: 30]
  --seed=<arg>                          Seed for the pseudo random number generator [default: 0]
  --database-original-directory=<arg>   Database original directory
  --database-extension=<arg>            Database extension [default: .jpg]
  -h --help                             Show this screen.
"""

from docopt import docopt
import logging
import bob.bio.base
import bob.core
import os
from bob.bio.htface.tools import FileSelector

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl
from sklearn.manifold import TSNE

logger = logging.getLogger("bob.bio.base")
from matplotlib import colors as mcolors
import bob.core
import bob.bio.base
from mpl_toolkits.mplot3d import Axes3D
numpy.random.seed(10)

from sklearn.preprocessing import normalize


def main():
    """Executes the main function"""

    args = docopt(__doc__, version='Run experiment')
    bob.core.log.set_verbosity_level(logger, 3)
    preprocessor_resource_key = "preprocessor"
    database_resource_key = "database"

    database_name = args['<database>']
    protocol = args['<protocol>']
    database_original_directory = args['--database-original-directory']
    database_extension = args['--database-extension']
    output_file = args['--output-file']

    seed = int(args['--seed'])
    perplexity = int(args['--perplexity'])
    learning_rate = float(args['--learning-rate'])
    iterations = int(args['--iterations'])

    # Loading the database
    database = bob.bio.base.utils.load_resource(database_name, database_resource_key,
                                                imports=['bob.bio.base'], package_prefix='bob.bio.',
                                                preferred_package=None)
    database.protocol = protocol

    FileSelector.create(
        database=database,
        modality_separator=database.modality_separator,
        extractor_file="",
        projector_file="",
        enroller_file="",

        preprocessed_directory="",
        extracted_directory="",
        projected_directory="",
        model_directories="",
        score_directories="",
        zt_score_directories="",
        compressed_extension="",
        default_extension='.hdf5',
    )

    fs = FileSelector.instance()

    logger.debug("  >> Loading data ...")
    original_datalist = fs.original_data_list(groups="world")
    
    data = []

    # Keeping two lists for each modality. Will be useful to use the colors
    indexes_modality = dict()
    indexes_modality[database.modalities[0]] = []
    indexes_modality[database.modalities[1]] = []
    #import bob.ip.color
    for o, i in zip(original_datalist, range(len(original_datalist))):

        if o.modality == database.modality_separator:
            indexes_modality[database.modalities[0]].append(i)
        else:
            indexes_modality[database.modalities[1]].append(i)
            
        raw_data = bob.io.base.load(o.make_path(database_original_directory) + database_extension)
        if raw_data.ndim==1:
            data.append(raw_data)
        else:
            #data.append(numpy.reshape(raw_data, (numpy.prod(raw_data.shape))))
            data.append(numpy.reshape(raw_data, (numpy.prod(raw_data.shape))))

    data = numpy.array(data)
    std = numpy.array([max(i,10e-6) for i in numpy.std(data, axis=0)])
    data = (data - numpy.mean(data, axis=0)) / std

    logger.debug("  >> Training TSNE with {0} ...".format(data.shape))
    model = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                 learning_rate=learning_rate, n_iter=iterations, init='pca', metric='euclidean', method='exact')
    projected_data = model.fit_transform(data)

    #pp = PdfPages(output_file)
    logger.debug("  >> Plotting projected data")
    fig = mpl.figure()
    #ax = mpl.subplot(111, projection='3d')
    ax = mpl.subplot(111)
    
    mpl.title("T-SNE - '{0}'".format(database_name))

    ax.scatter(projected_data[indexes_modality[database.modalities[0]], 0],
               projected_data[indexes_modality[database.modalities[0]], 1],
               c="cadetblue")

    ax.scatter(projected_data[indexes_modality[database.modalities[1]], 0],
               projected_data[indexes_modality[database.modalities[1]], 1],
               c="indianred")

    """
    ax.scatter(projected_data[indexes_modality[database.modality_separator], 0],
               projected_data[indexes_modality[database.modality_separator], 1],
               projected_data[indexes_modality[database.modality_separator], 2],
               c="cadetblue")

    ax.scatter(projected_data[indexes_modality["not_{0}".format(database.modality_separator)], 0],
               projected_data[indexes_modality["not_{0}".format(database.modality_separator)], 1],
               projected_data[indexes_modality["not_{0}".format(database.modality_separator)], 2],
               c="indianred")
    """

    mpl.legend(database.modalities)
    fig.savefig(output_file)

    #pp.savefig(fig)
    #pp.close()
    #del pp

    logger.debug("  >> Plot saved in '{0}'".format(output_file))
    logger.debug("  >> Done !!!")


if __name__ == "__main__":
    main()
