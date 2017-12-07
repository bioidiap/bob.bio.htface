#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  CUHK_CUFSF (with Feret) database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to CUHK_CUFSF database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from bob.bio.face.database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFile
import bob.io.base


class CUHK_CUFSFBioFile(FaceBioFile):

    def __init__(self, f, db):
        super(CUHK_CUFSFBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self.f = f
        self.db  = db

    def load(self, directory=None, extension=""):
        #if directory is None:
        return bob.io.base.load(self.db.original_file_name(self.f))
        #else:
        #    return self.f.load(directory=directory, extension=extension)

    @property
    def modality(self):
        return self.f.modality


class CUHK_CUFSFBioDatabase(BioDatabase):
    """
    Implements verification API for querying CUHK_CUFS database.
    """

    def __init__(
            self,
            original_directory="",
            original_extension = ['.jpg','.tif'],
            feret_directory="",
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(CUHK_CUFSFBioDatabase, self).__init__(name='cuhk_cufsf', **kwargs)

        from bob.db.cuhk_cufsf.query import Database as LowLevelDatabase
        self.db = LowLevelDatabase(original_directory=original_directory,
                                   original_extension=original_extension,
                                   feret_directory=feret_directory
                                   )

        self.original_directory = original_directory
        self.original_extension = original_extension
        self.feret_directory = feret_directory

                                   
    @property
    def modality_separator(self):
        return self.db.modality_separator

    @property
    def modalities(self):
        return self.db.modalities

    def original_file_name(self, file, check_existence = True):
        return self.db.original_file_name(file, check_existence = check_existence)    

    def model_ids_with_protocol(self, groups=None, protocol="search_split1_p2s", **kwargs):
        return self.db.model_ids(groups=groups, protocol=protocol)
        
    def objects(self, groups=None, protocol="search_split1_p2s", purposes=None, model_ids=None, **kwargs):
        retval = self.db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [CUHK_CUFSFBioFile(f, self.db) for f in retval]
        
    def annotations(self, file_object):
        return file_object.f.annotations()

