import pickle
import numpy as np
import cv2
from utils import download_weight

model_180 = download_weight('rotate_180.pkl')


def run(image):
	with open(model_180, 'rb') as f:
		model = pickle.load(f)
		img_arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		img_arr = cv2.resize(img_arr, (128,128))
		img_arr = np.array(img_arr).reshape(128*128)
		predicted = model.predict([img_arr])
		
		if predicted == 0:
			return cv2.rotate(image, cv2.ROTATE_180), 1
		
		return image, 0
