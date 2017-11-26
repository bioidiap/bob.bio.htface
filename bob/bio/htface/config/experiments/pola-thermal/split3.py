#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.htface.database import Pola_ThermalBioDatabase

#Applying the VGG crop in the casia webface DB
POLA_THERMAL_IMAGE_DIR = "/idiap/project/hface/databases/polimetric_thermal_database/Registered"

database = Pola_ThermalBioDatabase(original_directory=POLA_THERMAL_IMAGE_DIR,
                                   original_extension='.png',
                                   protocol='VIS-polarimetric-overall-split3',
                                  models_depend_on_protocol = True)

sub_directory = 'split3'

