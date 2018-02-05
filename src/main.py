import sys


from contrast_matrix import get_contrast_map_with_hist_info
from mean_shift_segmentation import segment


if __name__ == '__main__':
	original_image = cv2.imread(sys.argv[1])
	segmented_image = segment(original_image)
	contrast_map = get_contrast_map_with_hist_info(original_image)