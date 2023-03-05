import numpy as np
from skimage import io, transform
import os
from utils import PROGRESS

def load(data_folder, image_shape):
	data = []
	labels = []
	for foldername in os.listdir(data_folder):
		folder_path = f'{data_folder}/{foldername}'
		files = os.listdir(folder_path)
		prog_bar = PROGRESS(files)
		for filename in files:
			input_path = os.path.join(folder_path, filename)
			
			image = io.imread(input_path, True)
			
			image = transform.resize(image, image_shape)
			
			data.append(image)
			labels.append(foldername)
			
			prog_bar.update()
		
	return (np.array(data), np.array(labels))   
