import os
import time
import datetime
import numpy as np
from skimage import io
import cv2

def check_output_folder(folder):
	""" Create output folder if not exist - Tạo thư mục output nếu chưa tồn tại """
	if not os.path.exists(folder):
		os.mkdir(folder)


def output_exist(output_path):
	check = True if os.path.exists(output_path) else False
	return check


class PROGRESS:
	def __init__(self, i_list):
		self.__total = len(i_list)
		self.__current = 0
		self.__start = 0
		self.__mean = []

	def __per_sec(self):
		prev = time.time()
		ps = (1 / (prev - self.__start)) if self.__start != 0 else 0
		if len(self.__mean) >= 100:
			self.__mean.pop(0)
		self.__mean.append(ps)
		self.__start = prev
		return np.mean(self.__mean)

	def update(self, bar_length=50):
		self.__current += 1
		c = self.__current
		t = self.__total
		percent = c / t
		arrow = int(percent * bar_length) * '█'
		padding = int(bar_length - len(arrow)) * '.'
		ending = '\n' if c == t else '\r'
		ps = self.__per_sec()
		time_left = round((t - c) / ps) if ps != 0 else 0
		print(f'Progress: {c}/{t} |{arrow}{padding}| {int(percent * 100)}% in {datetime.timedelta(seconds=time_left)}s     ', end=ending)
		


def crop_background(image):
	if not type(image).__module__ == np.__name__:
		image = io.imread(image, True)
	if image.ndim > 2:
		gray = image[:,:,0]
	_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	x, y, w, h = cv2.boundingRect(thresholded)
	output = image[y:y+h, x:x+w]
	return output
			
