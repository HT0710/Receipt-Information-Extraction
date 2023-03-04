from skimage import io
import numpy as np

from rotation.CRAFT import model, net
from rotation.utils import rotate_box


def run(image, bboxes):
    if bboxes is not []:
        ratios = []
        for box in bboxes:
            x_min = min(box, key=lambda x: x[0])[0]
            x_max = max(box, key=lambda x: x[0])[0]
            y_min = min(box, key=lambda x: x[1])[1]
            y_max = max(box, key=lambda x: x[1])[1]
            if (x_max - x_min) > 20:
                ratio = (y_max - y_min) / (x_max - x_min)
                ratios.append(ratio)

        mean_ratio = np.mean(ratios)
        if mean_ratio >= 1:
            image, bboxes = rotate_box(image, bboxes, None, True, False)
            
    return image
