import argparse
from utils import load_config, crop_background, output_exist, measure, Progress
import cv2
import os
from time import time
from PIL import Image
from multiprocessing import Pool
import psutil
from rembg import remove
from rotation.CRAFT import model
from rotation.utils import Craft, align_box
from rotation import rotate_90, rotate_180
from text_extraction.vietocr import Config, Predictor
import numpy as np
import torch


class Pipeline:
	def __init__(self, config):
		self.config = config
		self.data = None
		self.text_detector = None
		self.text_extractor = None

	def load_image(self, image_path):
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		height = self.config['image_size']
		width = int(image.shape[1]*(height/image.shape[0]))
		return cv2.resize(image, (width, height))

	@measure
	def prepare_data(self):
		if not os.path.exists(self.config["input"]):
			print(f'[Error] No such file or directory: {self.config["input"]}')
			os._exit(0)
		if not output_exist(self.config["output"]):
			os.mkdir(self.config["output"])

		if os.path.isfile(self.config["input"]):
			filename = self.config["input"].split('/')[-1]
			self.data = [{
				'name': filename.split('.')[0],
				'image': self.load_image(self.config["input"])
				}]
			self.config["input"] = self.config["input"].replace(filename, '')
		else:
			files = os.listdir(self.config["input"])
			self.data = [{
				'name': filename.split('.')[0],
				'image': self.load_image(f'{self.config["input"]}/{filename}')
				} for filename in files]
		return self.data

	@staticmethod
	def remove_background(img_data):
		bg_removed = remove(img_data['image'])
		img_data['image'] = crop_background(bg_removed)
		return img_data

	@measure
	def prepare_model(self):
		ocr_config = Config.load_config_from_name(self.config['vietocr_model'])
		if (self.config['gpu'] != 0) and torch.cuda.is_available():
			self.text_detector = Craft('cuda')
			ocr_config['device'] = 'cuda' if self.config['gpu'] == -1 else f"cuda:{self.config['gpu']-1}"
		else:
			self.text_detector = Craft('cpu')
			ocr_config['device'] = 'cpu'
		self.text_extractor = Predictor(ocr_config)

	def rotate(self, img_data):
		img_1 = model.loadImage(img_data['image'])
		bboxes = self.text_detector(img_1)
		img_2 = rotate_90.run(img_1, bboxes)
		bboxes = self.text_detector(img_2)
		img_3, is_aligned = align_box(img_2, bboxes, skew_threshold=1)
		img_4, is_rotated = rotate_180.run(img_3)
		bboxes = self.text_detector(img_4) if (is_aligned or is_rotated) else bboxes
		img_data['image'] = img_4
		img_data['bboxes'] = bboxes
		return img_data

	def extract_info(self, img_data):
		img_data['information'] = []
		incline = {'prev_height': 0, 'prev_line': -1, }
		padding = round(self.config['image_size']*0.002)
		for i, box in enumerate(img_data['bboxes']):
			for px in [padding, padding*0.5, padding*0.25, 0]:
				x1 = int(box[0][0] - px)
				y1 = int(box[0][1] - px)
				x2 = int(box[2][0] + px)
				y2 = int(box[2][1] + px)
				arr_img = img_data['image'].copy()[y1:y2, x1:x2]  # crop image
				if 0 in [x for x in arr_img.shape]:
					continue

			try:
				img_box = Image.fromarray(arr_img)
				detected = self.text_extractor.predict(img_box)
			except:
				continue

			if self.config['save_box']:
				box_output = f"{self.config['output']}/{img_data['name']}"
				os.mkdir(box_output) if not output_exist(box_output) else None
				img_box.save(f'{box_output}/{i}.jpg')

			if self.config['incline']:
				current_height = sum(y[1] for y in box )/4
				per_diff = abs(1-incline['prev_height']/current_height)
				if per_diff < 0.015:
					img_data['information'][incline['prev_line']].append(detected)
				else:
					img_data['information'].append([detected])
					incline['prev_line'] += 1
				incline['prev_height'] = current_height
			else:
				img_data['information'].append([detected])
		return img_data

	def save_image(self, img_data):
		for box in img_data['bboxes']:
			poly = np.array(box).astype(np.int32).reshape((-1))
			poly = poly.reshape(-1, 2)
			image = np.ascontiguousarray(img_data['image'], dtype=np.uint8)
			cv2.polylines(image, [poly.reshape((-1, 1, 2))],
			              True, color=(0, 0, 255), thickness=2)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		Image.fromarray(image).save(f'{self.config["output"]}/{img_data["name"]}.jpg')

	def save_text(self, img_data):
		with open(f'{self.config["output"]}/{img_data["name"]}.txt', 'w+', encoding="utf-8") as f:
			for line in img_data['information']:
				f.write(' | '.join(line) + '\n')

def args_config(args):  # update default config with arguments
	config = load_config('config.yaml')['run']
	for arg, value in args.__dict__.items():
		if value is not None:
			config[arg] = value
	return config

@measure
def main(args):
	config = args_config(args)

	pl = Pipeline(config)

	data = pl.prepare_data()

	start = time()
	if config['multicore'] in [0, 1]:
		print(f'Multicore will not be used')
		bg_removed = [pl.remove_background(img_data) for img_data in data]
	else:
		max_cpu = psutil.cpu_count(logical=False)
		num_cpus = max_cpu if config['multicore'] == -1 else config['multicore']
		print(f'Maximum {num_cpus} cpu core will be used')
		with Pool(processes=num_cpus) as pool:
			bg_removed = pool.map(pl.remove_background, data)
	print(f'Done remove background in {round(time()-start, 2)}s')

	pl.prepare_model()
	
	print('Start extract info')
	for img_data in Progress(bg_removed):
		img_data = pl.rotate(img_data)
		img_data = pl.extract_info(img_data)
		pl.save_text(img_data) if config['save_text'] else None
		pl.save_image(img_data) if config['save_image'] else None
	print(f"Result has been saved to '{config['output']}'")


if __name__ == '__main__':
	args = argparse.ArgumentParser(description='Extract Receipt Information')  # setup execution argument
	args.add_argument('-i', '--input', type=str, help='Image path or Folder path (Default: data/test/)')
	args.add_argument('-o', '--output', type=str, help='Output folder path (Default: result/)')
	args.add_argument('-g', '--gpu', type=int, help='Use which gpu | 0 for cpu | -1 for all (Default: -1)')
	args.add_argument('-mc', '--multicore', type=int, help='Maximum of cpu core can use | -1 for all (Default: -1)')
	args = args.parse_args()

	main(args)
