#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import Pola_ThermalBioDatabase

POLA_THERMAL_IMAGE_DIR = "[YOUR_POLA_THERMAL_IMAGE_DIR]"

database = Pola_ThermalBioDatabase(original_directory=POLA_THERMAL_IMAGE_DIR,
                                original_extension='.png',
                                protocol='VIS-polarimetric-overall-split1',
                                models_depend_on_protocol = True)

