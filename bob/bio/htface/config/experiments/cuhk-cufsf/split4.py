#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 20 Oct 2016 16:48:32 CEST


import bob.bio.face
from bob.bio.htface.database import CUHK_CUFSFBioDatabase

#Applying the VGG crop in the casia webface DB
CUFSF_DATABASE_DIR = "/idiap/resource/database/CUHK-CUFSF/original_sketch"
FERET_DATABASE_DIR = "/idiap/temp/tpereira/databases/feret_cuhk-cufsf/feret_photos/feret"

database = CUHK_CUFSFBioDatabase(original_directory=CUFSF_DATABASE_DIR,
                                original_extension=['.jpg','.tif'],
                                feret_directory=FERET_DATABASE_DIR,
                                protocol='search_split4_p2s',
                                models_depend_on_protocol = True)


sub_directory = 'split4'
