from utils import check_output_folder, progress_bar
from PIL import Image, ImageOps
from rembg import remove
import os

raw_folder = 'data/raw' # Thư mục gốc
output_folder = 'data/background_removal' # Thư mục đầu ra

def main():
	check_output_folder(output_folder)
	
	for path, dirs, files in os.walk(raw_folder): # load thư mục gốc
		total = len(files)
		for i, file in enumerate(files, 1): # duyệt qua từng file trong thư mục
			input_path = f'{raw_folder}/{file}'
			input = Image.open(input_path) # đọc ảnh
			
			output = remove(input) # xóa nền ảnh
			output = ImageOps.grayscale(output) # chuyển sang trắng đen
			
			output_path = f'{output_folder}/{file}'
			output.save(output_path) # lưu ảnh đã xóa nền
			
			progress_bar(i, total)

if __name__ == '__main__':
	main()

