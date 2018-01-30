import cv2
import numpy as np
import sys
from mean_shift_segmentation import segment 

theta = 5

def rgb_to_gray(rgb_image):
	gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
	return gray_image

def add_padding(image):
	image = cv2.copyMakeBorder( image, theta, theta, theta, theta, cv2.BORDER_REPLICATE)
	return image

def get_neighbour_indices(cur_i,cur_j,theta=5):
	"""
	Returns tuple of tuples containing neighbours' indices
	Ex: ((0,0),(10,10))
	"""
	n1 = (cur_i - theta,cur_j - theta)
	n2 = (cur_i + theta,cur_j + theta)
	return (n1,n2)

def get_normalized_channels(image):
	norm_b =  cv2.normalize(image[:,:,0].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_g =  cv2.normalize(image[:,:,1].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_r =  cv2.normalize(image[:,:,2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	return np.asarray([norm_b,norm_g,norm_r])

def get_distance(image,i,j,n1,n2):
	d = 0.0
	for p in xrange(n1[0],n2[0]):
		for q in xrange(n1[1],n2[1]):
			# print image[:,i,j]-image[:,p,q]
			d += np.linalg.norm(image[:,i,j]-image[:,p,q])
	return d

def gaussian_blur(image):
	return cv2.GaussianBlur(image,(theta,theta),0)

def get_normalized_contrast_matrix(image):
	C = np.zeros(image.shape)
	seg = segment(image)
	cv2.imwrite("seg_blur.png",seg)
	seg = add_padding(seg)
	norm_seg = get_normalized_channels(seg)

	for i in xrange(theta,theta+shape[0]):
		for j in xrange(theta,theta+shape[1]):
			print i-theta,j-theta
			n1,n2 = get_neighbour_indices(i,j)
			C[i-theta,j-theta] = get_distance(norm_seg,i,j,n1,n2)

	norm_C = cv2.normalize(C.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	return norm_C
	


if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	#original_image = gaussian_blur(gaussian_blur(original_image))
	shape = original_image.shape
	norm_C_0 = get_normalized_contrast_matrix(original_image)
	blur_l1 = gaussian_blur(original_image)
	norm_C_1 = get_normalized_contrast_matrix(blur_l1)
	blur_l2 = gaussian_blur(blur_l1)
	norm_C_2 = get_normalized_contrast_matrix(blur_l2)
	
	norm_C_sum = norm_C_0 + norm_C_1 + norm_C_2

	norm_C_final = cv2.normalize(norm_C_sum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	contrast_matrix = np.round(norm_C_final*255)
	cv2.imwrite("contrast_blur.png",contrast_matrix)
	"""
	gray_seg = rgb_to_gray(segmented_image)
	float_gray_seg = cv2.normalize(gray_seg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	float_gray_seg = add_padding(float_gray_seg)

	#print max([max(row) for row in float_gray_seg])
	cv2.imwrite("gray_seg.png",gray_seg)
	"""