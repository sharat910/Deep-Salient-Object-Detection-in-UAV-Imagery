import cv2
from mean_shift_segmentation import segment 

theta = 5

def rgb_to_gray(rgb_image):
	gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
	return gray_image

def add_padding(image):
	image = cv2.copyMakeBorder( image, theta, theta, theta, theta, cv2.BORDER_REPLICATE)
	return image

def get_neighbour_indices(cur_i,cur_j,theta):
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
	return (norm_b,norm_g,norm_r)

#def euclidian_distance(image,i,j,p,q):


if __name__ == '__main__':
	original_image = cv2.imread("example.png")
	shape = original_image.shape
	print shape
	seg = segment(original_image)
	seg = add_padding(seg)
	norm_seg = get_normalized_channels(seg)

	for i in xrange(theta,theta+shape[0]):
		for j in xrange(theta,theta+shape[1]):
			print norm_seg[0][i,j]

	print seg.shape

	"""
	gray_seg = rgb_to_gray(segmented_image)
	float_gray_seg = cv2.normalize(gray_seg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	float_gray_seg = add_padding(float_gray_seg)

	#print max([max(row) for row in float_gray_seg])
	cv2.imwrite("gray_seg.png",gray_seg)
	"""