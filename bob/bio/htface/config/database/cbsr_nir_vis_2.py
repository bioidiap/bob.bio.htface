#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import CBSR_NIR_VIS_2BioDatabase

CBSR_IMAGE_DIR = "[YOUR_CBSR_IMAGE_DIR]"

database = CBSR_NIR_VIS_2BioDatabase(original_directory=CBSR_IMAGE_DIR,
                                original_extension=['.bmp', '.jpg'],
                                protocol='view2_1',
                                models_depend_on_protocol = True)

