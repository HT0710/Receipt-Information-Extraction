import os
from skimage import io
from rembg import remove
from utils import check_output_folder, PROGRESS, output_exist, crop_background

input_folder = 'data/raw'
output_folder = 'data/background_removed'


def main():
	check_output_folder(output_folder)
	
	files = os.listdir(input_folder)
	
	prog_bar = PROGRESS(files)
	for filename in files:
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not output_exist(output_path):
			img = io.imread(input_path)

			bg_removed = remove(img)
			
			output = crop_background(bg_removed)

			io.imsave(output_path, output[:,:,0])
		
		prog_bar.update()
		

if __name__ == '__main__':
	main()
