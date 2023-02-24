from rembg import remove
from PIL import Image
import os


raw_folder = 'raw' # Thư mục gốc
output_folder = 'background_remove' # Thư mục đầu ra (chưa có phải tạo)

for path, dirs, files in os.walk(raw_folder): # duyệt qua thư mục gốc
	for file in files: # duyệt qua từng file trong thư mục
		input_path = f'{raw_folder}/{file}'
		input = Image.open(input_path) # đọc ảnh
		
		name = file.split('.')[0] # xóa đuôi .jpg
		
		if not os.path.exists(output_folder): # check thư mục output có tồn tại ko
			os.mkdir(output_folder) # tạo thư mục

		output_path = f'{output_folder}/{name}.png'
		output = remove(input) # xóa nền ảnh

		output.save(output_path) # lưu ảnh đã xóa nền ra output folder
				
