import os
from skimage import io
from rembg import remove
from utils import Progress, crop_background, measure

input_folder = 'data/raw'
output_folder = 'data/background_removed'


@measure
def main():
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	
	files = os.listdir(input_folder)
	
	for filename in Progress(files):
		input_path = os.path.join(input_folder, filename)
		output_path = os.path.join(output_folder, filename)
		
		if not os.path.exists(output_path):
			img = io.imread(input_path)

			bg_removed = remove(img)
			
			output = crop_background(bg_removed, grayscale=True)

			io.imsave(output_path, output)
		

if __name__ == '__main__':
	main()
