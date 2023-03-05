import os
import time
import datetime
import numpy as np
from skimage import io
import cv2


def output_exist(output_path):
	"""Check if output is already exist"""
	return os.path.exists(output_path)


class PROGRESS:
	"""Progress bar"""
	bar_length = 50
	
	def __init__(self, i_list):
		self.total = len(i_list)
		self.current = 0
		self.__start = 0
		self.__recently = []
	
	@classmethod
	def set_bar_length(cls, length):
		"""Set new bar length (Default: 50)"""
		cls.bar_length = length

	def __per_sec(self):
		prev = time.time()
		ps = (1 / (prev - self.__start)) if self.__start != 0 else 0
		self.__recently.pop(0) if len(self.__recently) >= 100 else None
		self.__recently.append(ps) if ps <= 100 else None
		self.__start = prev
		return np.mean(self.__recently)
		
	def __time_calc(self):
		ps = self.__per_sec()
		sec_left = round((self.total - self.current) / ps) if ps != 0 else 0
		time_format = datetime.timedelta(seconds=sec_left)
		return time_format

	def __bar(self):
		percent = self.current / self.total
		completed = int(percent * PROGRESS.bar_length) * '█'
		padding = int(PROGRESS.bar_length - len(completed)) * '.'
		prog_line = f'Progress: {self.current}/{self.total} |{completed}{padding}| {int(percent*100)}% in {self.__time_calc()}s     '
		return prog_line
	
	def update(self):
		"""Finish current state and move to next state"""
		self.current += 1
		ending = '\n' if self.current == self.total else '\r'
		print(self.__bar(), end=ending)


def crop_background(image, grayscale=False):
	"""Crop black background only"""
	if not type(image).__module__ == np.__name__:
		img_arr = io.imread(image, True)
	else:
		img_arr = image
	gray = img_arr[:,:,0] if img_arr.ndim > 2 else img_arr
	_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	x, y, w, h = cv2.boundingRect(thresholded)
	output = (gray if grayscale else image)[y:y+h, x:x+w]
	return output
