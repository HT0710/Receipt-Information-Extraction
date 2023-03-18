import os
from skimage import io
from rembg import remove
from utils import output_exist, Progress, crop_background, measure

input_folder = 'data/raw'
output_folder = 'data/background_removed'


@measure
def main():
	if not output_exist(output_folder):
		os.mkdir(output_folder)
	
	files = os.listdir(input_folder)
	
	prog_bar = Progress(files)
	for filename in files:
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not output_exist(output_path):
			img = io.imread(input_path)

			bg_removed = remove(img)
			
			output = crop_background(bg_removed, grayscale=True)

			io.imsave(output_path, output)
		prog_bar.update()
		

if __name__ == '__main__':
	main()
