from skimage import io
from rotation import model, Craft
from text_extraction import Config, Predictor
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
	
	for box in Progress(bboxes):
		x1 = int(box[0][0] if (box[0][0] < box[3][0]) else box[3][0])
		y1 = int(box[0][1] if (box[0][1] < box[1][1]) else box[1][1])
		x2 = int(box[2][0] if (box[2][0] > box[1][0]) else box[1][0])
		y2 = int(box[2][1] if (box[2][1] > box[3][1]) else box[3][1])
		arr_img = _image.copy()[y1:y2, x1:x2] # crop image
		
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
