import numpy as np
import colorsys
import cv2
import sys
#from contrast_matrix import get_normalized_channels

def rgb_to_hsv_index(rgb_arr):
	hsv = colorsys.rgb_to_hsv(*rgb_arr)
	return np.round(255*np.asarray(hsv)).astype('int')


def get_normalized_channels(image):
	norm_b =  cv2.normalize(image[:,:,0].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_g =  cv2.normalize(image[:,:,1].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_r =  cv2.normalize(image[:,:,2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	return np.asarray([norm_b,norm_g,norm_r])

def get_hist(image_norm_channels):
	shape = image_norm_channels[0].shape
	hist_array = np.zeros((256,256))
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			hsv_index = rgb_to_hsv_index(image_norm_channels[:,i,j])
			hist_array[hsv_index[0],hsv_index[1]] += 1

	hist_array = cv2.normalize(hist_array.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	return hist_array

if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	image_chan = get_normalized_channels(original_image)
	histogram = get_hist(image_chan)
	for i in histogram:
		print max(i)