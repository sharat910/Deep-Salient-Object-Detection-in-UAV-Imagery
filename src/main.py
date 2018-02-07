import sys
import cv2
import numpy as np
from contrast_matrix import get_contrast_map_with_hist_info
from mean_shift_segmentation import segment
from edge_functions import get_final_salient_map
from color_channels import get_seven_normalized_color_channels

if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	cv2.imwrite("images/blue0.png",original_image[:,:,0])
	cv2.imwrite("images/green1.png",original_image[:,:,1])
	cv2.imwrite("images/red2.png",original_image[:,:,2])
	sys.exit()
	segmented_image = segment(original_image)
	#contrast_map = get_contrast_map_with_hist_info(original_image)
	contrast_map = np.zeros((original_image.shape[0],original_image.shape[1]))
	norm = get_seven_normalized_color_channels(original_image)
	salient_map = get_final_salient_map(segmented_image,contrast_map,norm)
	salient_map = np.round(salient_map*255)
	cv2.imwrite("images/salient_map.png",salient_map)