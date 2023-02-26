import numpy as np
import cv2


def rotate_box(img, bboxes, degree, rotate_90, flip):
    h, w = img.shape[:2]
    if degree:
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, degree)
        return new_img, np.array(new_bboxes)
    if rotate_90:
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return new_img, np.array(new_bboxes)

    if flip:
        new_bboxes = [[[w - i[0], h - i[1]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, cv2.ROTATE_180)
        return new_img, np.array(new_bboxes)
    return img, bboxes
