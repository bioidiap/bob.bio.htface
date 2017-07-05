#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  CUHK_CUFS database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to CUHK_CUFS database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from bob.bio.face.database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFile
import bob.io.base


class Pola_ThermalBioFile(FaceBioFile):

    def __init__(self, f):
        super(Pola_ThermalBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self.f = f

    @property
    def modality(self):
        return self.f.modality


class Pola_ThermalBioDatabase(BioDatabase):
    """
    Implements verification API for querying Pola_Thermal database.
    """

    def __init__(
            self,
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(Pola_ThermalBioDatabase, self).__init__(name='pola_thermal', **kwargs)

        from bob.db.pola_thermal.query import Database as LowLevelDatabase
        self.db = LowLevelDatabase()

    def model_ids_with_protocol(self, groups=None, protocol="VIS-polarimetric-overall-split1", **kwargs):
        return self.db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol="VIS-polarimetric-overall-split1", purposes=None, model_ids=None, **kwargs):
        retval = self.db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [Pola_ThermalBioFile(f) for f in retval]

    def annotations(self, file_object):
        return file_object.f.annotations()
        
