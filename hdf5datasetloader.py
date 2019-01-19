import h5py
import numpy as np


class Hdf5DatasetLoader:

    def load(self, db_path, max_items=np.inf):
        db = h5py.File(db_path)
        images = np.array(db["images"])
        measurements = np.array(db["measurements"])
        db.close()

        if max_items == np.inf or max_items > len(images):
            max_items = len(images)

        images = images[0:max_items]
        measurements = measurements[0:max_items]

        return images, measurements
