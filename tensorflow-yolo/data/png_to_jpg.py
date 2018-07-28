from PIL import Image
import glob
import os



file_list = glob.glob("images/*_co.png")

for file in file_list:
	im = Image.open(file)
	rgb_im = im.convert('RGB')
	rgb_im.save('images_jpg/%s.jpg' % (file.split("/")[1].split("_")[0]))
