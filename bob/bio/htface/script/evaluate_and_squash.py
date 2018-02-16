#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
This script runs CMC, DET plots and Recognition prints of groups of experiments.
It's useful when an evaluation protocol is based of k-fold cross validation.


Let's say you have executed 2 different experiments in a dataset whose protocol has five folds.
The command bellow will search for the scores of every fold and average them accordingly

Examples:

  `bob_htface_evaluate_and_squash.py <experiment_1> [<experiment_2>] --legends experiment1 --legends experiment2`
  

Usage:
  bob_htface_evaluate_and_squash.py  <experiment>... --legends=<arg>... [--title=<arg>] [--report-name=<arg>] [--colors=<arg>]... [--score-base-name=<arg>]
  bob_htface_evaluate_and_squash.py -h | --help

Options:
  --legends=<arg>                Name of each experiment  
  --colors=<arg>                 Colors of each plot
  --report-name=<arg>            Name of the report [default: report_name.pdf]
  --title=<arg>                  Title of the plot
  --score-base-name=<arg>        Name of the score files [default: scores-dev]
  -h --help                      Show this screen.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bob.measure
import numpy
import math
import os
from docopt import docopt

# matplotlib stuff

import matplotlib; matplotlib.use('agg') #avoids TkInter threaded start
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

# enable LaTeX interpreter
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('lines', linewidth = 4)

# increase the default font size
import bob.core
logger = bob.core.log.setup("bob.bio.base")


def _plot_cmc(cmcs, colors, labels, title, linestyle,  fontsize=12, position=None, xmin=0, xmax=100):

  if position is None: position = 4

  # open new page for current plot

  figure = pyplot.figure(dpi=600)
  offset = 0
  step   = int(len(cmcs)/len(labels))
  params = {'legend.fontsize': int(fontsize)}
  matplotlib.rcParams.update(params)
  matplotlib.rc('xtick', labelsize=18)
  matplotlib.rc('ytick', labelsize=18)
  matplotlib.rcParams.update({'font.size': 20})

  #For each group of labels
  max_x   =  0 #Maximum CMC size
  for i in range(len(labels)):
    #Computing the CMCs

    cmc_curves = []
    for j in range(offset,offset+step):
      cmc_curves.append(bob.measure.cmc(cmcs[j]))
      max_x = max(len(cmc_curves[j-offset]), max_x)

    #Adding the padding with '1's
    cmc_accumulator = numpy.zeros(shape=(step,max_x), dtype='float')
    for j in range(step):
      padding_diff =  max_x-len(cmc_curves[j])
      cmc_accumulator[j,:] = numpy.pad(cmc_curves[j],(0,padding_diff), 'constant',constant_values=(1))
      #cmc_average  += numpy.pad(cmc_curves[j],(0,padding_diff), 'constant',constant_values=(1))

    cmc_std     = numpy.std(cmc_accumulator, axis=0); cmc_std[-1]
    cmc_average = numpy.mean(cmc_accumulator, axis=0)

    if(linestyle is not None):    
      pyplot.semilogx(range(1, cmc_average.shape[0]+1), cmc_average * 100, lw=2, ms=10, mew=1.5, label=labels[i], ls=linestyle[i].replace('\\',''), color=colors[i])
    else:
      pyplot.semilogx(range(1, cmc_average.shape[0]+1), cmc_average * 100, lw=2, ms=10, mew=1.5, label=labels[i], color=colors[i])

    pyplot.errorbar(range(1, cmc_average.shape[0]+1), cmc_average*100, cmc_std*100, lw=0.5, ms=10,color=colors[i])
    offset += step    

  # change axes accordingly
  ticks = [int(t) for t in pyplot.xticks()[0]]
  pyplot.xlabel('Rank')
  pyplot.ylabel('Probability (\%)')
  pyplot.xticks(ticks, [str(t) for t in ticks])
  #pyplot.axis([0, max_x, xmin, 100])
  pyplot.axis([0, max_x, xmin, xmax])  
  pyplot.legend(loc=position)
  pyplot.title(title)
  pyplot.grid(True)

  return figure


def _compute_rr(cmcs, labels):
  offset = 0
  step   = int(len(cmcs)/len(labels))

  #Computing the recognition rate for each score file
  rr     = []   
  for i in range(len(cmcs)):
    rr.append(bob.measure.recognition_rate(cmcs[i]))
    
  average   = {}
  std_value = {}
  for i in range(len(labels)):
    l = labels[i]
    average   = round(numpy.mean(rr[offset : offset+step])*100,3)
    std_value = round(numpy.std(rr[offset : offset+step])*100,3)
    print("The AVERAGE Recognition Rate of the development set of '{0}' along '{1}' splits is {2}({3})".format(l, int(step), average, std_value))
    offset += step


def discover_scores(base_path, score_name="scores-dev", skip=["extracted", "preprocessed", "gridtk_logs"]):
    """
    Given a base path, get all the score files.
    """
    
    files = os.listdir(base_path)
    score_files = []
    for f in files:

        if f in skip:
            continue

        filename = os.path.join(base_path, f)
        if os.path.isdir(filename):
            score_files += discover_scores(filename, score_name)
        
        if f==score_name:
            score_files += [filename]
        
    return score_files


def main(command_line_parameters=None):
    """Reads score files, computes error measures and plots curves."""

    args = docopt(__doc__, version='Run experiment')


    # check that the legends have the same length as the dev-files
    if (len(args["<experiment>"]) % len(args["--legends"])) != 0:
        logger.error("The number of experiments (%d) is not multiple of --legends (%d) ", len(args["<experiment>"]), len(args["--legends"]))

    
    bob.core.log.set_verbosity_level(logger, 3)
    dev_files = []
    for e in args["<experiment>"]:
        df = discover_scores(e, score_name=args["--score-base-name"])
        dev_files += df
        logger.info("{0} scores discovered in {1}".format(len(df), e))
    
    # RR
    logger.info("Computing recognition rate")
    cmcs_dev = [bob.measure.load.cmc_four_column(f) for f in dev_files]
    _compute_rr(cmcs_dev, args["--legends"])
    
    # CMC
    logger.info("Plotting CMC")
    if len(args["--colors"]) ==0:
        colors     = ['red','green','blue', 'black','cyan', 'magenta', 'yellow']
    else:
        if (len(args["<experiment>"]) % len(args["--colors"])) != 0:
            logger.error("The number of experiments (%d) is not multiple of --colors (%d) ", len(args["<experiment>"]), len(args["--colors"]))
    
    pdf = PdfPages(args["--report-name"])
    try:
        # create a separate figure for dev and eval
        fig = _plot_cmc(cmcs_dev, colors, args["--legends"], args["--title"], linestyle=None)
        pdf.savefig(fig)    
  
    except RuntimeError as e:
        raise RuntimeError("During plotting of CMC curves, the following exception occured:\n%s\nUsually this happens when the label contains characters that LaTeX cannot parse." % e)

    pdf.close()
    logger.info("Done !!!")


if __name__ == '__main__':
    main()

