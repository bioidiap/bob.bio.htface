from .baseline import Baseline
from .databases import *
from inception_v2 import *

def get_all_baselines():

    baselines = dict()
    for baseline in Baseline.__subclasses__():
        b = baseline()
        baselines[b.name] = b
        
    return baselines


def get_all_databases():

    databases = dict()
    for database in Databases.__subclasses__():
        d = database()
        databases[d.name] = d
        
    return databases
        

def get_config():
  """Returns a string containing the configuration information.
  """

  import bob.extension
  return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
