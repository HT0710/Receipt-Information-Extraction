import cv2
import numpy as np
import imutils	
import math

from rotation.CRAFT import net


model, refine_net, cuda = net.setup()
def craft(image):
	bboxes, _, _ = net.test_net(model, image, 0.7, 0.4, 0.4, cuda, False, refine_net)
	return bboxes
	

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


def align_box(image, bboxes, skew_threshold=5, top_box=3):
	vertical_vector = [0, -1]
	top_box = np.argpartition([box[1][0]- box[0][0] for box in bboxes], -top_box)[-top_box:]
	avg_angle = 0
	for idx in top_box:
		skew_vector = bboxes[idx][0] - bboxes[idx][3]
		angle = np.math.atan2(np.linalg.det([vertical_vector,skew_vector]),np.dot(vertical_vector,skew_vector))
		avg_angle += math.degrees(angle)/3

	if abs(avg_angle) < skew_threshold:
		return image, 0
	return imutils.rotate(image, avg_angle), 1
