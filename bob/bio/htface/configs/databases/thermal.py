#!/usr/bin/env python

import bob.bio.face
from bob.bio.htface.database import Pola_ThermalBioDatabase

#database = Pola_ThermalBioDatabase(original_directory=thermal["data_path"],
#                                original_extension=thermal["extension"],
#                                protocol='VIS-thermal-overall-split1',
#                                models_depend_on_protocol = True)


database = Pola_ThermalBioDatabase(original_directory="",
                                original_extension="",
                                protocol='VIS-thermal-overall-split1',
                                models_depend_on_protocol = True)

# Estimated training size
samples_per_epoch = 400 * 5
