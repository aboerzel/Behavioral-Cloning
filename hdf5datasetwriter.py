import os
import h5py


class HDF5DatasetWriter:
    def __init__(self, images_dims, measurement_dims, outputPath, bufSize=1000):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied `outputPath` already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)

        outputDir = os.path.dirname(outputPath)
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # open the HDF5 database for writing and create two floyd:
        # one to store the images and another to store the measurements
        self.db = h5py.File(outputPath, "w")
        self.images = self.db.create_dataset("images", images_dims, dtype="uint8")
        self.measurements = self.db.create_dataset("measurements", measurement_dims, dtype=float)

        # store the buffer size, then initialize the buffer itself
        # along with the index into the floyd
        self.bufSize = bufSize
        self.buffer = {"images": [], "measurements": []}
        self.idx = 0

    def add(self, images, measurements):
        # add the images and measurements to the buffer
        self.buffer["images"].extend(images)
        self.buffer["measurements"].extend(measurements)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["images"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["images"])
        self.images[self.idx:i] = self.buffer["images"]
        self.measurements[self.idx:i] = self.buffer["measurements"]
        self.idx = i
        self.buffer = {"images": [], "measurements": []}

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["images"]) > 0:
            self.flush()

        # close the images
        self.db.close()
