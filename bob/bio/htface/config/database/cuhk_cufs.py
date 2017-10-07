#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CUHK_CUFSBioDatabase

CUFS_DATABASE_DIR = "[CUFS_DATABASE_DIR]"
ARFACE_DATABASE_DIR = "[ARFACE_DATABASE_DIR]"
XM2VTS_DATABASE_DIR = "[XM2VTS_DATABASE_DIR]"

database = CUHK_CUFSBioDatabase(cufs_database_dir=CUFS_DATABASE_DIR,
                                arface_database_dir=ARFACE_DATABASE_DIR,
                                xm2vts_database_dir=XM2VTS_DATABASE_DIR,
                                protocol='search_split1_p2s',
                                models_depend_on_protocol = True
                                )


