#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CBSR_NIR_VIS_2BioDatabase

database = CBSR_NIR_VIS_2BioDatabase(original_directory=casia_nir_vis["data_path"],
                                original_extension=casia_nir_vis["extension"],
                                protocol='view2_1',
                                models_depend_on_protocol = True)

