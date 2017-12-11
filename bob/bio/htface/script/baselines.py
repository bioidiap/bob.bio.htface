#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
This script runs some face recognition baselines under some face databases

Run `bob_htface_baselines.py  --list` to list all the baselines and databases available

Examples:

This command line will run the facenet from David Sandberg using the IJB-A dataset:
  `bob_htface_baselines.py --baselines facenet_msceleba_inception_v1 --databases ijba`
  

This command line will run all the registed baselines using all databases registered databases::
  `bob_htface_baselines.py --baselines all --databases all`



Usage:
  bob_htface_baselines.py  --baselines=<arg> --databases=<arg>  [--protocol=<arg>]
  bob_htface_baselines.py  --list
  bob_htface_baselines.py -h | --help

Options:
  --baselines=<arg>                   Baseline name [default: all]
  --databases=<arg>                   Database name [default: all]
  --list                              List all the registered baselines
  -h --help                           Show this screen.
  -p, --protocol=<arg>                Protocol name  (If not set, all the protocols from a particular database will be triggered)
"""


import bob.bio.base
import bob.io.base
import os
from bob.extension.config import load
import pkg_resources
from docopt import docopt
from bob.bio.base.script.verify import main as verify


base_paths = pkg_resources.resource_filename("bob.bio.htface",
                                             "configs/base_paths.py")

from .registered_baselines import all_baselines, resources


def trigger_verify(preprocessor, extractor, database, groups, sub_directory, protocol=None,
                   preprocessed_directory=None, extracted_directory=None, random_config_file_name=None):
    
    configs  = load([base_paths])    
    
    parameters = [
        base_paths,
        random_config_file_name,
        database,
        preprocessor,
        extractor,
        '-a', "distance-cosine",
        '-g', 'demanding',
        '-vvv',
        '--temp-directory', configs.temp_dir,
        '--result-directory', configs.results_dir,
        '--sub-directory', sub_directory,
        '--environment','LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0',
    ] + ['--groups'] + groups

    if protocol is not None:
        parameters += ['--protocol', protocol]

    if preprocessed_directory is not None:
        parameters += ['--preprocessed-directory', preprocessed_directory]

    if extracted_directory is not None:
        parameters += ['--extracted-directory', extracted_directory]

    return parameters


def write_protocol(file_name, protocol):
    """
    Writes the protocol in a file which will be one of the config files chain loaded
    
    Parameters
    ----------
      file_name:
        Name of the temporary config file
        
      protocol:
        Name of the protocol
    
    """
    
    open(file_name, "w").write("protocol=\"%s\""%protocol)
    

def run_cnn_baseline(baseline, databases=resources["databases"].keys(), reuse_extractor=False, protocol=None):
    configs  = load([base_paths])
        
    for d in databases:        
        first_protocol = resources["databases"][d]["protocols"][0]
        first_subdir = os.path.join(d, resources[baseline]["name"], first_protocol)
        
        extracted_directory = None
        if ("reuse_extractor" in resources[baseline] and
           resources[baseline]["reuse_extractor"]):

            extracted_directory=os.path.join(configs.temp_dir, first_subdir, "extracted")
        
        # Iterate over the protocols
        if protocol is None:
            protocols = resources["databases"][d]["protocols"]
        else:
            protocols = [protocol]
            
        for p in protocols:
        
            import tensorflow as tf
            tf.reset_default_graph()

            sub_directory = os.path.join(d, resources[baseline]["name"], p)

            # Writing file to ship the protocol name to be chain loaded
            protocol_config_file_name = os.path.join(configs.temp_dir, sub_directory,p + ".py")
            bob.io.base.create_directories_safe(os.path.dirname(protocol_config_file_name))
            write_protocol(protocol_config_file_name, p)
            
            parameters = trigger_verify(resources[baseline]["preprocessor"],
                                        resources[baseline]["extractor"],
                                        resources["databases"][d]["config"],
                                        resources["databases"][d]["groups"],
                                        sub_directory,
                                        protocol=p,
                                        preprocessed_directory=os.path.join(configs.temp_dir, first_subdir, "preprocessed"),
                                        extracted_directory=extracted_directory,
                                        random_config_file_name=protocol_config_file_name
                                        )
            verify(parameters)


def main():

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
        for b in all_baselines:
            run_cnn_baseline(baseline=b, database=database, protocol=args["--protocol"])
    else:
        run_cnn_baseline(baseline=args["--baselines"], databases=database, protocol=args["--protocol"])


if __name__ == "__main__":
    main()