from skimage import io
from rotation.CRAFT import model
from rotation.utils import craft
from text_extraction.vietocr import Config, Predictor
from PIL import Image

# load vietocr
config = Config.load_config_from_name("vgg_seq2seq")
config['device'] = 'cuda:0'  # config['device'] = 'cpu'
detector = Predictor(config)


def main():
	img_path = 'data/rotated/bhx_5bcac72d2869ee37b778.jpg'

	_image = io.imread(img_path)

	bboxes = craft(model.loadImage(_image)) # craft
	
	prev_height = 0
	prev_line = -1
	informations = []
	
	for i, box in enumerate(bboxes, 1):
		for padding in [10, 5, 0]:
			x1 = int(box[0][0]-padding)
			y1 = int(box[1][1]-padding)
			x2 = int(box[2][0]+padding)
			y2 = int(box[2][1]+padding)
			
			arr_img = _image.copy()[y1:y2, x1:x2] # crop image
			if not 0 in [x for x in arr_img.shape]:
				break
		
		image = Image.fromarray(arr_img)
		
		detect = detector.predict(image)
		
		current_height = sum(y[1] for y in box )/4
		if abs(1-prev_height/current_height) < 0.02:
			informations[prev_line].append(detect)
		else:
			informations.append([detect])
			prev_line += 1
		prev_height = current_height
		
	for i in informations:
		print(i)


if __name__ == '__main__':
	main()
