from bob.bio.htface.datashuffler import SiameseDiskHTFace
import bob.io.base
import bob.io.image

def test_siamease_datashuffler_cuhk_cufs():


    from bob.db.cuhk_cufs.query import Database
    database = Database(original_directory="/idiap/temp/tpereira/HTFace/CUHK-CUFS/SIAMESE/split1/preprocessed/",
                        original_extension=".hdf5",
                        arface_directory="", xm2vts_directory="")

    siamese_disk_htface = SiameseDiskHTFace(database=database, protocol="search_split1_p2s",
                                            batch_size=8,
                                           input_shape=[None, 224, 224, 1])

    offset = 0
    import ipdb; ipdb.set_trace();
    while True:
        print offset
        offset += 1
        batch = siamese_disk_htface.get_batch()
        if siamese_disk_htface.epoch > 1:
            break

    assert True
