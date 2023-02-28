import os
import time

import numpy as np


def check_output_folder(folder):
    """ Create output folder if not exist - Tạo thư mục output nếu chưa tồn tại """
    if not os.path.exists(folder):
        os.mkdir(folder)


class PROGRESS:
    def __init__(self):
        self.__start = 0
        self.__mean = []

    def per_sec(self):
        prev = time.time()
        ps = (1 / (prev - self.__start)) if self.__start != 0 else 0
        if len(self.__mean) >= 10:
            self.__mean.pop(0)
        self.__mean.append(ps)
        self.__start = prev
        return np.mean(self.__mean)

    def show(self, current, total, bar_length=50):
        fraction = current / total
        arrow = int(fraction * bar_length) * '█'
        padding = int(bar_length - len(arrow)) * '.'
        ending = '\n' if current == total else '\r'
        ps = self.per_sec()
        time_left = round((total - current) / ps) if ps != 0 else 0
        print(f'Progress: {current}/{total} |{arrow}{padding}| {int(fraction * 100)}% in {time_left}s     ', end=ending)


def output_exist(input_path, output_path):
    check = True if os.path.exists(f'{output_path}/{input_path}') else False
    return check
