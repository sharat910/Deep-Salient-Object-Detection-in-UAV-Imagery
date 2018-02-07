import numpy as np 
import cv2


def gradient(img):
	kernel = np.ones((5,5), np.uint8)

	img_erosion = cv2.erode(img, kernel, iterations=1)
	img_dilation = cv2.dilate(img, kernel, iterations=1)

	return (img_dilation - img_erosion)


def canny(img):
	return cv2.Canny(img,100,200)

def get_final_salient_map(seg_image,contrast_map,norm_channels):
	seg_canny = canny(seg_image)
	print "Seg canny shape:",seg_canny.shape
	canny_contrast = canny(np.uint8(contrast_map))
	seven_canny = map(canny,map(lambda x: np.uint8(np.round(x*255)),norm_channels))

	final_canny = seg_canny + canny_contrast + sum(seven_canny)
	final_canny = cv2.normalize(final_canny.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	

	seg_gradient = gradient(seg_image)
	seg_gradient = np.sum(seg_gradient,axis=2)
	print "Seg gradient shape:",seg_gradient.shape

	gradient_contrast = gradient(contrast_map)
	seven_gradient = map(gradient,map(lambda x: np.uint8(np.round(x*255)),norm_channels))

	final_gradient = seg_gradient + gradient_contrast + sum(seven_gradient)
	final_gradient = cv2.normalize(final_gradient.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	
	return np.multiply(final_canny,final_canny)