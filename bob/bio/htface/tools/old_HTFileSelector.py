#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Sun Mar  07 17:15:00 CET 2015


from bob.bio.base import utils
from bob.bio.base.tools.FileSelector import FileSelector


@utils.Singleton
class HTFileSelector(FileSelector):
  """This class provides shortcuts for selecting different files for different stages of the verification process"""

  def __init__(self, separator,  
      # parameters of the File selector
      **kwargs):

    super(HTFileSelector, self).__init__(**kwargs)
    self.m_separator = separator
    

  def __arrange_by_client(self, files, directory_type):
    return [self.get_paths(files[client], directory_type) for client in range(len(files))]
    
    
  def __arrange_by_modality(self, files, directory_type):

    #lists for each modality
    list_A = []
    list_B = []
      
    #group per modality
    for f in files:
      
      if type(f) is str:
        if f.find(self.m_separator) > -1:
          list_A.append(f)
        else:
          list_B.append(f)
          
      else:

        if f.path.find(self.m_separator) > -1:
          list_A.append(self.get_paths([f],directory_type)[0])
        else:
          list_B.append(self.get_paths([f],directory_type)[0])

    return [list_A,list_B]


  ### Training lists
  def training_list(self, directory_type, step, arrange_by_client = False, arrange_by_modality=False):
    """Returns the list of features that should be used for projector training.
    The directory_type might be any of 'preprocessed', 'features', or 'projected'.
    The step might by any of 'train_extractor', 'train_projector', or 'train_enroller'.
    If arrange_by_client is enabled, a list of lists (one list for each client) is returned."""

    files = self.m_database.training_files(step, arrange_by_client)

    if not arrange_by_client and not arrange_by_modality:
      return files

    elif not arrange_by_client and arrange_by_modality:
      return self.__arrange_by_modality(files, directory_type)

    elif arrange_by_client and not arrange_by_modality:
      return self.__arrange_by_client(files, directory_type)
      
    else: #arrange_by_client and arrange_by_modality:
      clients = self.__arrange_by_client(files, directory_type)
      list_A = []
      list_B = []
      
      for c in clients:
        a,b = self.__arrange_by_modality(c, directory_type)
        list_A.append(a)
        list_B.append(b)
        
      return [list_A, list_B]
        

