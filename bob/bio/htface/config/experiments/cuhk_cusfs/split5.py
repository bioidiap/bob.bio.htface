#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.htface.database import CUHK_CUFSBioDatabase

#Applying the VGG crop in the casia webface DB
CUFS_DATABASE_DIR = "/idiap/resource/database/CUHK-CUFS"
ARFACE_DATABASE_DIR = "/idiap/resource/database/AR_Face/images"
XM2VTS_DATABASE_DIR = "/idiap/resource/database/xm2vtsdb/images"

database = CUHK_CUFSBioDatabase(cufs_database_dir=CUFS_DATABASE_DIR,
                                arface_database_dir=ARFACE_DATABASE_DIR,
                                xm2vts_database_dir=XM2VTS_DATABASE_DIR,
                                protocol='search_split5_p2s',
                                models_depend_on_protocol = True)

sub_directory = 'split5'

