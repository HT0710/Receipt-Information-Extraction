import pickle
import numpy as np
from skimage import transform, color

with open('weights/rotate_180.pkl', 'rb') as f:
    model = pickle.load(f)


def run(image):
	img_arr = color.rgb2gray(image)
	img_arr = transform.resize(img_arr, (128,128))
	img_arr = np.array(img_arr).reshape(128*128)
	predicted = model.predict([img_arr])
	
	if predicted == 0:
		image = transform.rotate(image, 180)
		
	f.close()
	
	return image
