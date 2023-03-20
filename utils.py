import os
from time import time
import datetime
import numpy as np
from skimage import io
import cv2
import yaml
from functools import wraps


class Progress:
	"""Progress bar"""
	def __init__(self, i_list):
		self.__list = i_list
		self.total = len(i_list)
		self.current = -1
		self.__bar_length = 0
		self.__begin_time = time()
		self.__start = 0
		self.__recently = []
	
	def __iter__(self):
		return self
		
	def __next__(self):
		self.__update()
		if self.current < self.total:
			return self.__list[self.current]
		raise StopIteration
	
	def __update_bar_length(self, bar):
		terminal_lenght = os.get_terminal_size()[0]
		external_bar_length = int(len(bar)-self.__bar_length)
		self.__bar_length = terminal_lenght-external_bar_length-2

	def __per_sec(self):
		prev = time()
		ps = (1 / (prev - self.__start)) if self.__start != 0 else 0
		self.__recently.pop(0) if len(self.__recently) >= 100 else None
		self.__recently.append(ps) if ps <= 100 else None
		self.__start = prev
		return np.mean(self.__recently)
		
	def __time_left(self):
		ps = self.__per_sec()
		sec_left = round((self.total - self.current) / ps) if ps != 0 else 0
		time_format = datetime.timedelta(seconds=sec_left)
		return time_format

	def __time_total(self):
		sec_total = round(time()-self.__begin_time)
		time_format = datetime.timedelta(seconds=sec_total)
		return time_format

	def __bar(self):
		percent = self.current / self.total
		completed = int(percent * self.__bar_length) * '█'
		padding = int(self.__bar_length - len(completed)) * '.'
		return f'Progress: {self.current}/{self.total} |{completed}{padding}| {int(percent*100)}% in {self.__time_left()}s > {self.__time_total()}s '
	
	def __update(self):
		"""Finish current state and move to next state"""
		self.current += 1
		bar = self.__bar()
		ending = '\n' if self.current == self.total else '\r'
		self.__update_bar_length(bar)
		print(bar, end=ending)


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


def load_config(config_name, args=None):
	"""Load config file"""
	with open('config.yaml') as f:
		config = yaml.full_load(f)[config_name]
	if args is not None:
		for arg, value in args.__dict__.items():
			if value is not None:
				config[arg] = value
	return config
	

def measure(func):
	"""Measure the runtime"""
	@wraps(func)
	def _time(*args, **kwargs):
		start = time()
		try:
			return func(*args, **kwargs)
		finally:
			end_ = time() - start
			print(f"Done {func.__name__} in {round(end_, 2)}s")
	return _time