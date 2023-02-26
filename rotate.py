from utils import check_output_folder, progress_bar, output_exist
from rotation import rotate_90
import os


cuda = True
raw_folder = 'data/background_removed'
output_folder = 'data/rotated'


def main():
    check_output_folder(output_folder)

    for path, dirs, files in os.walk(raw_folder):
        total = len(files)
        for i, file in enumerate(files, 1):
            if not output_exist(file, output_folder):
                input_path = f'{raw_folder}/{file}'

                rotate_90.run(input_path, output_folder, cuda)

            progress_bar(i, total)


if __name__ == '__main__':
    main()
