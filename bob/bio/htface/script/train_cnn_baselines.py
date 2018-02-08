#!/usr/bin/env python

"""Trains the HTFace baselines using Tensorflow estimators.

Run `bob_htface_cnn_baselines.py  --list` to list all the baselines and databases available

Examples:

This command line will run the facenet from David Sandberg using the IJB-A dataset:
  `bob_htface_cnn_baselines.py --baselines facenet_msceleba_inception_v1 --databases ijba`
  

This command line will run all the registed baselines using all databases registered databases::
  `bob_htface_cnn_baselines.py --baselines all --databases all`



Usage:
  bob_htface_cnn_baselines.py  --baselines=<arg> --databases=<arg> [-v ...] [--protocol=<arg>]
  bob_htface_cnn_baselines.py  --list
  bob_htface_cnn_baselines.py -h | --help


Options:
    -h --help                          Show this help message and exit
    -v, --verbose                      Increases the output verbosity level
    -p, --protocol=<arg>               Protocol name
    

The configuration files should have the following objects totally:

  ## Required objects:

  estimator
  train_input_fn

  ## Optional objects:

  hooks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
from bob.bio.base.utils import read_config_file
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)

import os
import bob.io.base
from docopt import docopt
from .baselines import write_protocol
from bob.bio.htface.baselines import get_all_baselines, get_all_databases, get_all_baselines_by_type


def train_cnn(baseline, database, protocol, args):
    """
    Trains the CNN given
    """
    
    # 1 - Paths config file for the chain loading
    config_base_path = pkg_resources.resource_filename("bob.bio.htface",
                                                 "configs/base_paths.py")

    # 2 - Database preprocessed paths
    config_preprocessing = os.path.join(baseline.preprocessed_data, database.name+".py")

    # 3 - Protocol Hack
    configs  = read_config_file([config_base_path])
    config_protocol_file_name = os.path.join(configs.temp_dir, "protocols_tmp", baseline.name, protocol+".py")
    bob.io.base.create_directories_safe(os.path.dirname(config_protocol_file_name))
    write_protocol(config_protocol_file_name, protocol)
    
    # 4 - Database obj
    config_database = database.config
    
    # 5 - Estimator
    config_estimator = baseline.estimator

    config = read_config_file([config_base_path, config_preprocessing, config_protocol_file_name,
                                config_database, config_estimator])
    
    hooks = getattr(config, 'hooks', None)
    
    # Sets-up logging
    set_verbosity_level(logger, args["--verbose"])

    # required arguments
    estimator = config.estimator
    train_input_fn = config.train_input_fn

    # Train
    estimator.train(input_fn=train_input_fn, hooks=hooks,
                    steps=config.steps)


def main(argv=None):

    all_baselines = get_all_baselines()
    all_databases = get_all_databases()
    all_baselines_by_type = get_all_baselines_by_type()

    args = docopt(__doc__, version='Run experiment')
    if args["--list"]:
        print("====================================")
        print("Follow all the registered baselines:")
        print("====================================")        
        for a in all_baselines_by_type:
            print("Baselines of the type: %s"%(a))
            for b in all_baselines_by_type[a]:
                print("  >> %s"%(b))
            print("\n")

        print("====================================")
        print("Follow all the registered databases:")
        print("====================================")
        for a in all_databases:
            print("  - %s"%(a))
        print("\n")
        exit()
    
    if args["--databases"] == "all":
        database = all_databases.keys()
    else:
        database = [all_databases[args["--databases"]]]

    if args["--baselines"] == "all":
        baselines = all_baselines.keys()
    else:
        baselines = [ all_baselines[ args["--baselines"] ] ]

    # Triggering training for each baseline/database/protocol
    for b in baselines:
        for d in database:
            if args["--protocol"] is None:
                for p in all_databases[d].protocols:
                    train_cnn(baseline=b, database=d, protocol=p, args=args)
            else:
                train_cnn(baseline=b, database=d, protocol=args["--protocol"], args=args)

if __name__ == '__main__':
    main()
