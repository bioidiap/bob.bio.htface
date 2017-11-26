#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 20 Oct 2016 16:48:32 CEST


import bob.bio.face
from bob.bio.htface.database import CBSR_NIR_VIS_2BioDatabase

#Applying the VGG crop in the casia webface DB
CBSR_IMAGE_DIR = "[YOUR_CBSR_IMAGE_DIR]"

database = CBSR_NIR_VIS_2BioDatabase(original_directory=CBSR_IMAGE_DIR,
                                original_extension=['.bmp', '.jpg'],
                                protocol='view2_3',
                                models_depend_on_protocol = True)

sub_directory = 'split3'

