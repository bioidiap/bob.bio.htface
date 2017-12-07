#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import random
import sys
import os
import bob.io.base
def create_random_name(base_path):
    """
    Creates a random name used to name files
    """
    if sys.version_info[0] == 2:
      from string import letters as ascii_letters
    else:
      from string import ascii_letters

    bob.io.base.create_directories_safe(base_path)
    return os.path.join(base_path, "".join(random.sample(ascii_letters, 10))) + ".py"



def get_cnn_model_name(base_path, baseline_name, database_name, protocol):
    """
    Given a database and protocol, creates the cnn model name
    
    Parameters
    ----------
    
    base_path:
    
    baseline_name:
    
    database_name:
    
    protocol:
    
    """
    
    return os.path.join(base_path, "cnn", baseline_name, database_name, protocol)

