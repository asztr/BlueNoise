#import pyexr
import numpy as np
import os.path as op
import cv2
import warnings

#for file reading
def gamma_decode(rgb, gamma=2.2, maxval=255.0):
	return (rgb/maxval)**gamma

#for file writing
def gamma_encode(rgb, gamma=2.2, maxval=255.0):
	return maxval*(rgb**(1.0/gamma))

def img_f2i(img_f):
	return (img_f*255).astype(np.uint8)

def img_i2f(img_i):
	return img_i.astype(np.float32)/255.0

def read_image(fname, uint8=False):
	ext = op.splitext(fname)[-1].lower()
	img_i = cv2.imread(fname)
	img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
	img_f = img_i2f(img_i)
	if uint8 is True:
		return img_i
	return img_f

def write_image(fname, img):
	ext = op.splitext(fname)[-1].lower()
	if img.dtype == np.uint8:
		img_i = img
	else:
		img_i = img_f2i(img)
	cv2.imwrite(fname, cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
