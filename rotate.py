import os

from rotation import rotate_90
from utils import check_output_folder, PROGRESS, output_exist

raw_folder = 'data/background_removed'
output_folder = 'data/rotated'


def main():
    check_output_folder(output_folder)

    bar = PROGRESS()
    for path, dirs, files in os.walk(raw_folder):
        total = len(files)
        for i, file in enumerate(files, 1):
            if output_exist(file, output_folder):
                continue

            input_path = f'{raw_folder}/{file}'

            rotate_90.run(input_path, output_folder)

            bar.show(i, total)


if __name__ == '__main__':
    main()
