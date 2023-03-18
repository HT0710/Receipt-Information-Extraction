import pickle
import numpy as np
import cv2

with open('weights/rotate_180.pkl', 'rb') as f:
    model = pickle.load(f)


def run(image):
	img_arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	img_arr = cv2.resize(img_arr, (128,128))
	img_arr = np.array(img_arr).reshape(128*128)
	predicted = model.predict([img_arr])
	
	if predicted == 0:
		return cv2.rotate(image, cv2.ROTATE_180), 1
	
	return image, 0
