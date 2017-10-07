#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CUHK_CUFSFBioDatabase

#bob.db.cuhk_cufsf.Database(original_directory='/idiap/resource/database/CUHK-CUFSF/original_sketch/', original_extension=['.jpg','.tif'], feret_directory='/idiap/temp/tpereira/databases/#feret_cuhk-cufsf/feret_photos/feret/')

CUFSF_DATABASE_DIR = "[CUFSF_DATABASE_DIR]"
FERET_DATABASE_DIR = "[FERET_DATABASE_DIR"

database = CUHK_CUFSBioDatabase(original_directory=CUFS_DATABASE_DIR,
                                original_extension=['.jpg','.tif'],
                                feret_directory=FERET_DATABASE_DIR,
                                protocol='search_split1_p2s',
                                models_depend_on_protocol = True
                                )


