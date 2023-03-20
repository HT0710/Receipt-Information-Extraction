from skimage import io
from rotation.CRAFT import model
from rotation.utils import Craft
from text_extraction.vietocr import Config, Predictor
from PIL import Image
from utils import measure, Progress

# load craft
craft = Craft('cuda')  # 'cpu' to use cpu

# load vietocr
config = Config.load_config_from_name("vgg_seq2seq")
config['device'] = 'cuda:0'  # config['device'] = 'cpu'
detector = Predictor(config)


@measure
def main():
	img_path = 'data/rotated/bhx_5bcac72d2869ee37b778.jpg'

	_image = io.imread(img_path)

	bboxes = craft(model.loadImage(_image)) # craft
	
	prev_height = 0
	prev_line = -1
	information = []
	
	padding = round(_image.shape[1]*0.002)
	for box in Progress(bboxes):
		for px in [padding, padding*0.5, padding*0.25, 0]:
			x1 = int(box[0][0]-px)
			y1 = int(box[0][1]-px)
			x2 = int(box[2][0]+px)
			y2 = int(box[2][1]+px)
			
			arr_img = _image.copy()[y1:y2, x1:x2] # crop image
			if 0 in [x for x in arr_img.shape]:
				continue
		
		try:
			image = Image.fromarray(arr_img)
			detect = detector.predict(image)
		except:
			continue
		
		current_height = sum(y[1] for y in box )/4
		if abs(1-prev_height/current_height) < 0.015:
			information[prev_line].append(detect)
		else:
			information.append([detect])
			prev_line += 1
		prev_height = current_height
		
	for i in information:
		print(i)


if __name__ == '__main__':
	main()
