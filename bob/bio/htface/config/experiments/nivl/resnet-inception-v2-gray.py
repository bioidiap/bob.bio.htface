#!/bin/bash
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

# Base settings

temp_directory         = "/idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/"
result_directory       = "/idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/" 
algorithm              = "distance-cosine"
env                    = ["LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0/bin"]
preprocessed_directory = "/idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split1/preprocessed"
extracted_directory    = "/idiap/temp/tpereira/HTFace/NIVL/idiap_inception_v2_gray--casia/split1/extracted"

