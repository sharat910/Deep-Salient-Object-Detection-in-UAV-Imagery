import csv
from collections import Counter

f = open("annotation1024.csv",'r')
reader = csv.reader(f)
c = Counter()

for row in reader:
	c[row[0]] += 1

print(c)
