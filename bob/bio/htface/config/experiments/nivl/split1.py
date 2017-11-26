#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.htface.database import NIVLBioDatabase

NIVL_IMAGE_DIR = "[YOUR_NIVL_IMAGE_DIR]"

database = NIVLBioDatabase(original_directory=NIVL_IMAGE_DIR,
                           original_extension='.png',
                           protocol='idiap-search_VIS-NIR_split1',
                           models_depend_on_protocol = True)

sub_directory = 'split1'

