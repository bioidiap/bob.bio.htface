#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

# Temp directories
temp_dir    = "/idiap/temp/tpereira/HTFace/"
results_dir = "/idiap/temp/tpereira/HTFace/"


# Datsbase paths
casia_nir_vis = {"data_path": "/idiap/resource/database/cbsr_nir_vis_2",
                 "inception_resnet_v2_path":"",
                 "extension": ['.bmp', '.jpg']}
                  
polathermal = {"data_path": "/idiap/project/hface/databases/polimetric_thermal_database/Registered",
               "inception_resnet_v2_path":"/idiap/temp/tpereira/HTFace/pola_thermal/idiap_casia_inception_v2_gray/VIS-polarimetric-overall-split1/preprocessed/",
               "extension": ".png"}

nivl = {"data_path": "/idiap/resource/database/nivl/nivl-dataset-v1.0",
        "inception_resnet_v2_path":"",
        "extension": ".png"}

cuhk_cufs = {"cufs_path": "/idiap/resource/database/CUHK-CUFS",
             "arface_path": "/idiap/resource/database/AR_Face/images",
             "xm2vts_path": "/idiap/resource/database/xm2vtsdb/images",
             "inception_resnet_v2_path":"",             
             "extension": ".png"}
            
cuhk_cufsf = {"cufsf_path": "/idiap/resource/database/CUHK-CUFSF/original_sketch/",
              "feret_path": "/idiap/project/hface/databases/feret_cuhk-cufsf/feret_photos/feret/",
              "inception_resnet_v2_path": "/idiap/temp/tpereira/HTFace/cuhk_cufsf/idiap_casia_inception_v2_gray/search_split1_p2s/preprocessed/",
              "extension": ['.jpg','.tif']}
            


# Background model paths

## STUFF BASED ON CASIA WEBFACE ##
inception_resnet_v2_casia_webface_gray = "/idiap/temp/tpereira/casia_webface/new_tf_format/official_checkpoints/inception_resnet_v2_gray/centerloss_alpha-0.95_factor-0.02_lr-0.1/"

