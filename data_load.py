import numpy as np
from skimage import io, transform
import os

class DatasetLoader:
    def load(self, image_paths, image_shape):

        # Initialize the list of images and labels
        data = []
        labels = []

        # Loop over input paths to read the data
        for (i, path) in enumerate(os.listdir(image_paths)):
            # Load images
            # Assuming path in following format
            input_path = os.path.join(image_paths, path)
            
            image = io.imread(input_path, True)

            
            label = path.split("_")[0]
            

            # Resize image
            image = transform.resize(image, image_shape)

            # Push into data list
            data.append(image)
            labels.append(label)

        # Return a tuple of data and labels
        return (np.array(data), np.array(labels))
