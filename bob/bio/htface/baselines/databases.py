#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pkg_resources

class Databases(object):
    """
    Baseclass for all the database resources for this project
    
    The database should have:
      - Name
      - Config
      - Protocol
      - Groups
    """
    def __init__(self):
        self.name = ""
        self.config = ""
        self.protocols = []
        self.groups = []


class CUHK_CUFS(Databases):

    def __init__(self):
        self.name = "cuhk_cufs"
        self.config = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/cuhk_cufs.py")
        self.protocols = ["search_split1_p2s", "search_split2_p2s", "search_split3_p2s", "search_split4_p2s", "search_split5_p2s"]
        self.groups = ["dev"]
    
    
class CUHK_CUFSF(Databases):

    def __init__(self):
        self.name = "cuhk_cufsf"
        self.config = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/cuhk_cufsf.py")
        self.protocols = ["search_split1_p2s", "search_split2_p2s", "search_split3_p2s", "search_split4_p2s", "search_split5_p2s"]
        self.groups = ["dev"]
    

class NIVL(Databases):

    def __init__(self):
        self.name = "nivl"
        self.config = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/nivl.py")
        self.protocols = ["idiap-search_VIS-NIR_split1", "idiap-search_VIS-NIR_split2", 
                          "idiap-search_VIS-NIR_split3", "idiap-search_VIS-NIR_split4", "idiap-search_VIS-NIR_split5"]
        self.groups = ["dev"]

    
class Casia_nir_vis(Databases):

    def __init__(self):
        self.name      = "casia_nir_vis"
        self.config    = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/casia_nir_vis.py")
        self.protocols = ["view2_1", "view2_2","view2_3","view2_4","view2_5"]
        self.groups    = ["eval"]


class Polathermal(Databases):

    def __init__(self):
        self.name      = "pola_thermal"
        self.config    = pkg_resources.resource_filename("bob.bio.htface", "configs/databases/pola_thermal.py")
        self.protocols = ["VIS-polarimetric-overall-split1", "VIS-polarimetric-overall-split2", "VIS-polarimetric-overall-split3",
                         "VIS-polarimetric-overall-split4", "VIS-polarimetric-overall-split5"]
        self.groups    = ["dev"]
    

