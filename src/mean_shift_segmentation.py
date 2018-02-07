import cv2
import sys
import pymeanshift as pms


def segment(original_image):
	(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, 
	                                                              range_radius=4.5, min_density=1)
	return segmented_image




if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])

	print original_image.shape

	segmented_image = segment(original_image)
	seg_edges = cv2.Canny(segmented_image,100,200)
	 
	# print segmented_image.shape
	# print labels_image,number_regions
	cv2.imwrite("images/seg_edges.png",seg_edges)