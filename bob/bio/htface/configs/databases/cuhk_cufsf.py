#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CUHK_CUFSFBioDatabase

database = CUHK_CUFSFBioDatabase(original_directory=cuhk_cufsf["cufsf_path"],
                                original_extension=cuhk_cufsf["extension"],
                                feret_directory=cuhk_cufsf["feret_path"],
                                protocol='search_split1_p2s',
                                models_depend_on_protocol = True
                                )

