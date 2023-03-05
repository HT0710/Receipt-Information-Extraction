import os
from skimage import io
from rotation import rotate_90, rotate_180
from utils import output_exist, PROGRESS, crop_background
from rotation.CRAFT import model
from rotation.utils import craft, align_box


input_folder = 'data/background_removed'
output_folder = 'data/rotated'


def main():
	if not output_exist(output_folder):
		os.mkdir(output_folder)

	files = os.listdir(input_folder)
	
	prog_bar = PROGRESS(files)
	for filename in files:
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not output_exist(output_path):
			img_0 = io.imread(input_path)
			
			img_1 = model.loadImage(img_0) # pre-format image

			bboxes = craft(img_1)
			img_2 = rotate_90.run(img_1, bboxes)

			bboxes = craft(img_2)
			img_3, is_align = align_box(img_2, bboxes, skew_threshold=1)
			
			img_4 = rotate_180.run(img_3)

			output = crop_background(img_4, grayscale=True)

			io.imsave(output_path, output)

		prog_bar.update()


if __name__ == '__main__':
	main()
