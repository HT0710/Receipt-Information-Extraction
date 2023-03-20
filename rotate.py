import os
from skimage import io
from utils import Progress, crop_background, measure
from rotation import model, Craft, align_box, rotate_90, rotate_180

input_folder = 'data/background_removed'
output_folder = 'data/rotated'
craft = Craft('cuda')  # 'cpu' to use cpu


@measure
def main():
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	files = os.listdir(input_folder)
	
	for filename in Progress(files):
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not os.path.exists(output_path):
			img_0 = io.imread(input_path)
			
			img_1 = model.loadImage(img_0)  # load image for craft

			bboxes = craft(img_1)
			img_2 = rotate_90.run(img_1, bboxes)  # rotated 90

			bboxes = craft(img_2)
			img_3, is_aligned = align_box(img_2, bboxes, skew_threshold=1)
			
			img_4, is_rotated = rotate_180.run(img_3)  # rotated 180

			output = crop_background(img_4, grayscale=True)

			io.imsave(output_path, output)


if __name__ == '__main__':
	main()
