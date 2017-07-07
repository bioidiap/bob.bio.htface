from bob.bio.htface.datashuffler import SiameseDiskHTFace


def test_siamease_datashuffler_cuhk_cufs():


    from bob.db.cuhk_cufs.query import Database
    database = Database(original_directory="/Users/tiago.pereira/Documents/database/cuhk_cufs_process",
                        original_extension=".png",
                        arface_directory="", xm2vts_directory="")

    siamese_disk_htface = SiameseDiskHTFace(database=database, protocol="cuhk_p2s",
                                            batch_size=8,
                                            input_shape=[None, 80, 64, 1])

    xxx = siamese_disk_htface.get_batch()
    import ipdb; ipdb.set_trace()

    assert True