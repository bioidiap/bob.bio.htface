#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import NIVLBioDatabase

database = NIVLBioDatabase(original_directory=nivl["data_path"],
                                original_extension=nivl["extension"],
                                protocol='idiap-search_VIS-NIR_split1',
                                models_depend_on_protocol = True)

