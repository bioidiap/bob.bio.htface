#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import Pola_ThermalBioDatabase

database = Pola_ThermalBioDatabase(original_directory=polathermal["data_path"],
                                original_extension=polathermal["extension"],
                                protocol='VIS-polarimetric-overall-split1',
                                models_depend_on_protocol = True)

