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
from .registered_baselines import all_baselines, resources
from docopt import docopt
from .baselines import write_protocol

def train_cnn(baseline, database, protocol, args):
    """
    Trains the CNN given
    """
    
    # 1 - Paths config file for the chain loading
    config_base_path = pkg_resources.resource_filename("bob.bio.htface",
                                                 "configs/base_paths.py")

    # 2 - Database preprocessed paths
    config_preprocessing = os.path.join(resources[baseline]["preprocessed_data"], database+".py")

    # 3 - Protocol Hack
    configs  = read_config_file([config_base_path])
    config_protocol_file_name = os.path.join(configs.temp_dir, "protocols_tmp", resources[baseline]["name"], protocol+".py")
    bob.io.base.create_directories_safe(os.path.dirname(config_protocol_file_name))
    write_protocol(config_protocol_file_name, protocol)
    
    # 4 - Database obj
    config_database = resources["databases"][database]["config"]
    
    # 5 - Estimator
    config_estimator = resources[baseline]["estimator"]

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


    args = docopt(__doc__, version='Run experiment')
    if args["--list"]:
        print("====================================")
        print("Follow all the registered baselines:")
        print("====================================")        
        for a in all_baselines:
            print("  - %s"%(a))
        print("\n")

        print("====================================")
        print("Follow all the registered databases:")
        print("====================================")
        for a in resources["databases"]:
            print("  - %s"%(a))
        print("\n")


        exit()
    
    if args["--databases"] == "all":
        database = resources["databases"].keys()
    else:
        database = [args["--databases"]]

    if args["--baselines"] == "all":
        baselines = resources["--baselines"].keys()
    else:
        baselines = [args["--baselines"]]

    # Triggering training for each baseline/database/protocol
    for b in baselines:
        for d in database:
            if args["--protocol"] is None:
                for p in resources["databases"][d]["protocols"]:
                    train_cnn(baseline=b, database=d, protocol=p, args=args)
            else:
                train_cnn(baseline=b, database=d, protocol=args["--protocol"], args=args)

if __name__ == '__main__':
    main()
