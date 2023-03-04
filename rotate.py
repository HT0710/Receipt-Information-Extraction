import os
from skimage import io
from rotation import rotate_90
from utils import check_output_folder, PROGRESS, output_exist, crop_background
from rotation.CRAFT import model
from rotation.utils import craft, align_box


input_folder = 'data/background_removed'
output_folder = 'data/rotated'


def main():
	check_output_folder(output_folder)

	files = os.listdir(input_folder)
	
	prog_bar = PROGRESS(files)
	for filename in files:
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not output_exist(output_path):
			img_0 = io.imread(input_path)
			
			img_1 = model.loadImage(img_0) # refine image

			bboxes = craft(img_1)
			img_2 = rotate_90.run(img_1, bboxes)

			bboxes = craft(img_2)
			img_3, is_align = align_box(img_2, bboxes, skew_threshold=0)
			
			#bboxes = craft(img_3) if is_align else bboxes
			#img_4 = rotate_180.run(img_3, bboxes)

			output = crop_background(img_3, grayscale=True)

			io.imsave(output_path, output)

		prog_bar.update()


if __name__ == '__main__':
	main()
