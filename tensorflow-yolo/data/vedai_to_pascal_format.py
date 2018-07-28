import csv,os

DATA_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(DATA_ROOT, 'images_jpg')

f = open("annotation1024.csv",'r')
reader = csv.reader(f)

wf = open("vedai.txt",'w')

line = ""
last_image_no = None
for row in reader:
	current_image = row[0]
	current_image_no = current_image.split("_")[0]
	if current_image_no != last_image_no:
		wf.write("%s\n"%line)
		line = DATA_PATH + "/%s.jpg" % current_image_no
	line += " "
	line += " ".join(row[4:])
	line += " 0"
	last_image_no = current_image_no
