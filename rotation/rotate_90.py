from skimage import io
import numpy as np

from rotation.CRAFT import model, net
from rotation.utils import rotate_box


def run(image_path, output_folder):
    image = model.loadImage(image_path)

    craft, cuda = net.net_setup()
    bboxes, _, _ = net.test_net(craft, image, 0.7, 0.4, 0.4, cuda, False, None)

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

    filename = image_path.split('/')[-1]
    io.imsave(f"./{output_folder}/{filename}", image[:,:,0])
