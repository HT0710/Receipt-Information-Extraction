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
	def __init__(self, args, config):
		self.input = args.input
		self.output = args.output
		self.size = config['image_size']
		self.data = {
			'input': [],
			'output': []
		}
		self.gpu = args.gpu
		self.text_detector = None
		self.extractor_model = config['vietocr_model']
		self.text_extractor = None
		self.incline = config['incline']

	@measure
	def prepare(self):
		if not os.path.exists(self.input):
			print(f"[Error] No such file or directory: {self.input}")
			os._exit(0)

		if os.path.isfile(self.input):
			input_path = os.path.join(self.input)
			output_path = os.path.join(self.output, self.input.split('/')[-1].split('.')[0])
			self.data['input'].append(input_path)
			self.data['output'].append(output_path)
			return self.data
		
		files = os.listdir(self.input)
		for filename in files:
			input_path = os.path.join(self.input, filename)
			output_path = os.path.join(self.output, filename.split('.')[0])
			self.data['input'].append(input_path)
			self.data['output'].append(output_path)
		return self.data

	def load_image(self, image_path):
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		width = int(image.shape[1]*(self.size/image.shape[0]))
		return cv2.resize(image, (width, self.size))

	def remove_background(self, image):
		bg_removed = remove(image)
		return crop_background(bg_removed)

	@measure
	def gpu_config(self):
		ocr_config = Config.load_config_from_name(self.extractor_model)
		if (self.gpu != 0) and torch.cuda.is_available():
			self.text_detector = Craft('cuda')
			ocr_config['device'] = 'cuda' if self.gpu == -1 else f'cuda:{self.gpu-1}'
		else:
			self.text_detector = Craft('cpu')
			ocr_config['device'] = 'cpu'
		self.text_extractor = Predictor(ocr_config)

	def rotate(self, image):
		craft = self.text_detector
		img_1 = model.loadImage(image)
		bboxes = craft(img_1)
		img_2 = rotate_90.run(img_1, bboxes)
		bboxes = craft(img_2)
		img_3, is_aligned = align_box(img_2, bboxes, skew_threshold=1)
		bboxes = craft(img_3) if is_aligned else bboxes
		img_4, is_rotated = rotate_180.run(img_3)
		bboxes = craft(img_4) if is_rotated else bboxes
		return img_4, bboxes

	def extract_info(self, _image, bboxes):
		vietocr = self.text_extractor
		inline = {'prev_height': 0, 'prev_line': -1, }
		informations = []
		padding = round(self.size*0.002)
		for i, box in enumerate(bboxes):
			for px in [padding, padding*0.5, padding*0.25, 0]:
				x1 = int(box[0][0] - px)
				y1 = int(box[0][1] - px)
				x2 = int(box[2][0] + px)
				y2 = int(box[2][1] + px)
				arr_img = _image.copy()[y1:y2, x1:x2]  # crop image
				if not 0 in [x for x in arr_img.shape]:
					break
			
			img_box = Image.fromarray(arr_img)
			detected = vietocr.predict(img_box)

			if self.incline:
				current_height = sum(y[1] for y in box )/4
				per_diff = abs(1-inline['prev_height']/current_height)
				if per_diff < 0.015:
					informations[inline['prev_line']].append(detected)
				else:
					informations.append([detected])
					inline['prev_line'] += 1
				inline['prev_height'] = current_height
			else:
				informations.append([detected])
		return informations

	def save_image(self, image, bboxes, output):
		if not output_exist(self.output):
			os.mkdir(self.output)
		for box in bboxes:
			poly = np.array(box).astype(np.int32).reshape((-1))
			poly = poly.reshape(-1, 2)
			image = np.ascontiguousarray(image, dtype=np.uint8)
			cv2.polylines(image, [poly.reshape((-1, 1, 2))],
			              True, color=(0, 0, 255), thickness=2)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		Image.fromarray(image).save(output + '.jpg')

	def save_text(self, output, infomation):
		if not output_exist(self.output):
			os.mkdir(self.output)
		with open(f'{output}.txt', 'w+', encoding="utf-8") as f:
			for line in infomation:
				f.write(' | '.join(line) + '\n')


@measure
def main(args):
	pl = Pipeline(args, cfg)

	data = pl.prepare()

	start = time()
	if args.multicore in [0, 1]:
		print(f'Multicore will not be used')
		bg_rv_imgs = []
		for image_path in data['input']:
			load_img = pl.load_image(image_path)
			bg_rv_imgs.append(pl.remove_background(load_img))
	else:
		num_cpus = psutil.cpu_count(logical=False) if args.multicore == -1 else args.multicore
		print(f'Maximum {num_cpus} cpu core will be used')
		with Pool(processes=num_cpus) as pool:
			load_imgs = pool.map(pl.load_image, data['input'])
			bg_rv_imgs = pool.map(pl.remove_background, load_imgs)
	print(f'Done remove background in {round(time()-start, 2)}s')

	pl.gpu_config()
	
	print('Start extract info')
	prog_bar = Progress(bg_rv_imgs)
	for output, image in zip(data['output'], bg_rv_imgs):
		rotated, bboxes = pl.rotate(image)
		info = pl.extract_info(rotated, bboxes)
		pl.save_image(rotated, bboxes, output) if cfg['save_image'] else None
		pl.save_text(output, info) if cfg['save_text'] else None
		prog_bar.update()
	print(f"Result has been saved to '{args.output}'")

if __name__ == '__main__':
	cfg = load_config('config.yaml')['run']

	args = argparse.ArgumentParser(description='Extract Receipt Information')  # setup execution argument
	args.add_argument('-i', '--input', default=cfg['input'], type=str, help='Image path or Folder path (Default: data/test/)')
	args.add_argument('-o', '--output', default=cfg['output'], type=str, help='Output folder path (Default: Output/)')
	args.add_argument('-g', '--gpu', default=cfg['gpu'], type=int, help='Use which gpu | 0 for cpu | -1 for all (Default: -1)')
	args.add_argument('-mc', '--multicore', default=cfg['multicore'], type=int, help='Maximum of cpu core can use | -1 for all (Default: -1)')
	args = args.parse_args()
	
	main(args)
