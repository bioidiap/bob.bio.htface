#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.face.database import FaceBioFile
from .cuhk_cufs import CUHK_CUFSBioDatabase
from .cuhk_cufsf import CUHK_CUFSFBioDatabase
from .nivl import NIVLBioDatabase
from .cbsr_nir_vis_2 import CBSR_NIR_VIS_2BioDatabase
from .pola_thermal import Pola_ThermalBioDatabase
from .pericrosseye import PeriCrossEyeBioDatabase


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
  """Says object was actually declared here, and not in the import module.
  Fixing sphinx warnings of not being able to find classes, when path is shortened.
  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

  for obj in args: obj.__module__ = __name__

__appropriate__(
    FaceBioFile,
    CUHK_CUFSBioDatabase,
    CUHK_CUFSFBioDatabase,
    NIVLBioDatabase,
    CBSR_NIR_VIS_2BioDatabase,
    Pola_ThermalBioDatabase,
    PeriCrossEyeBioDatabase
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]