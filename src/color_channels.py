import numpy as np
import cv2,sys
from edge_functions import canny,gradient

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
			S = float(sum(original_image[i,j,:])) / 3.0
			norm_b[i,j] = original_image[i,j,0] / S
			norm_g[i,j] = original_image[i,j,1] / S
			norm_r[i,j] = original_image[i,j,2] / S
			broad_b[i,j] = original_image[i,j,0] - (original_image[i,j,1] + original_image[i,j,2] / 2.0)
			broad_g[i,j] = original_image[i,j,1] - (original_image[i,j,0] + original_image[i,j,2] / 2.0)
			broad_r[i,j] = original_image[i,j,2] - (original_image[i,j,0] + original_image[i,j,1] / 2.0)
			broad_y[i,j] = ((original_image[i,j,2] + original_image[i,j,1]) \
								- abs(original_image[i,j,2] - original_image[i,j,1])) / 2.0


	norm_b = cv2.normalize(norm_b.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_g = cv2.normalize(norm_g.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	norm_r = cv2.normalize(norm_r.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_b = cv2.normalize(broad_b.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_g = cv2.normalize(broad_g.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_r = cv2.normalize(broad_r.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	broad_y = cv2.normalize(broad_y.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	#Testing
	#broad_y	= np.zeros((shape[0],shape[1])).astype('float')
	
	return (norm_b,norm_g,norm_r,broad_b,broad_g,broad_r,broad_y)


if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	norm = get_seven_normalized_color_channels(original_image)
	norm_canny = map(canny,map(lambda x: np.uint8(np.round(x*255)),norm))
	norm_gradient = map(gradient,map(lambda x: np.uint8(np.round(x*255)),norm))
	# cv2.imwrite("images/channel_blue.png",np.round(norm[0]*255))
	# cv2.imwrite("images/channel_green.png",np.round(norm[1]*255))
	# cv2.imwrite("images/channel_red.png",np.round(norm[2]*255))
	# cv2.imwrite("images/broad_blue.png",np.round(norm[3]*255))
	# cv2.imwrite("images/broad_green.png",np.round(norm[4]*255))
	# cv2.imwrite("images/broad_red.png",np.round(norm[5]*255))
	# cv2.imwrite("images/broad_yellow.png",np.round(norm[6]*255))

	# cv2.imwrite("images/channel_blue_canny.png",norm_canny[0])
	# cv2.imwrite("images/channel_green_canny.png",norm_canny[1])
	# cv2.imwrite("images/channel_red_canny.png",norm_canny[2])
	# cv2.imwrite("images/broad_blue_canny.png",norm_canny[3])
	# cv2.imwrite("images/broad_green_canny.png",norm_canny[4])
	# cv2.imwrite("images/broad_red_canny.png",norm_canny[5])
	# cv2.imwrite("images/broad_yellow_canny.png",norm_canny[6])

	cv2.imwrite("images/channel_blue_gradient.png",norm_gradient[0])
	cv2.imwrite("images/channel_green_gradient.png",norm_gradient[1])
	cv2.imwrite("images/channel_red_gradient.png",norm_gradient[2])
	cv2.imwrite("images/broad_blue_gradient.png",norm_gradient[3])
	cv2.imwrite("images/broad_green_gradient.png",norm_gradient[4])
	cv2.imwrite("images/broad_red_gradient.png",norm_gradient[5])
	cv2.imwrite("images/broad_yellow_gradient.png",norm_gradient[6])