import numpy as np
import cv2,sys

def get_seven_normalized_color_channels(original_image):
	'''
	Input w X h X 3
	'''
	shape = original_image.shape
	
	norm_b = np.zeros((shape[0],shape[1]))
	norm_g = np.zeros((shape[0],shape[1]))
	norm_r = np.zeros((shape[0],shape[1]))

	broad_b = np.zeros((shape[0],shape[1]))
	broad_g = np.zeros((shape[0],shape[1]))
	broad_r = np.zeros((shape[0],shape[1]))
	broad_y	= np.zeros((shape[0],shape[1]))
	
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			S = float(sum(original_image[i,j,:]))
			norm_b[i,j] = original_image[i,j,0] / S
			norm_g[i,j] = original_image[i,j,1] / S
			norm_r[i,j] = original_image[i,j,2] / S
			broad_b[i,j] = original_image[i,j,0] - (original_image[i,j,1] + original_image[i,j,2] / 2.0)
			broad_g[i,j] = original_image[i,j,1] - (original_image[i,j,0] + original_image[i,j,2] / 2.0)
			broad_r[i,j] = original_image[i,j,2] - (original_image[i,j,0] + original_image[i,j,1] / 2.0)
			broad_y[i,j] = ((original_image[i,j,2] + original_image[i,j,1]) \
								- abs(original_image[i,j,2] - original_image[i,j,1])) / 2.0


	broad_b = cv2.normalize(broad_b.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_g = cv2.normalize(broad_g.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_r = cv2.normalize(broad_r.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_y = cv2.normalize(broad_y.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	return (norm_b,norm_g,norm_r,broad_b,broad_g,broad_r,broad_y)


if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	norm = get_seven_normalized_color_channels(original_image)
	
	cv2.imwrite("channel_blue.png",np.round(norm[0]*255))
	cv2.imwrite("channel_green.png",np.round(norm[1]*255))
	cv2.imwrite("channel_red.png",np.round(norm[2]*255))
	cv2.imwrite("broad_blue.png",np.round(norm[3]*255))
	cv2.imwrite("broad_green.png",np.round(norm[4]*255))
	cv2.imwrite("broad_red.png",np.round(norm[5]*255))
	cv2.imwrite("broad_yellow.png",np.round(norm[6]*255))