import os


def check_output_folder(folder):
    """ Create output folder if not exist - Tạo thư mục output nếu chưa tồn tại """
    if not os.path.exists(folder):
        os.mkdir(folder)


def progress_bar(current, total, bar_length=100):
    """ Progress bar - Thanh tiến trình """
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress {current}/{total} : [{arrow}{padding}] {int(fraction * 100)}%', end=ending)


def output_exist(input_path, output_path):
    return True if os.path.exists(f'{output_path}/{input_path}') else False
