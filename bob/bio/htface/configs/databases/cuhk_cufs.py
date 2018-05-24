#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CUHK_CUFSBioDatabase

#database = CUHK_CUFSBioDatabase(cufs_database_dir=cuhk_cufs["cufs_path"],
#                                arface_database_dir=cuhk_cufs["arface_path"],
#                                xm2vts_database_dir=cuhk_cufs["xm2vts_path"],
#                                protocol='search_split1_p2s',
#                                original_extension=cuhk_cufs["extension"],
#                               models_depend_on_protocol = True
#                                )

database = CUHK_CUFSBioDatabase(cufs_database_dir="",
                                arface_database_dir="",
                                xm2vts_database_dir="",
                                protocol='search_split1_p2s',
                                original_extension="",
                                models_depend_on_protocol = True
                                )


# Estimated training size
samples_per_epoch = 404 * 5
