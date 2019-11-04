import os

dataset_root = "/path/to/dataset"

img_list = []

def check_num_images():
	jpg_count = 0
	gt_count = 0

	for root, dirs, files in os.walk(dataset_root):
		for fname in files:
			if 'jpg' in fname:
				jpg_count+=1
				img_list.append(fname[:-4])
			if 'png' in fname:
				gt_count+=1

	print ("num of images: {}, num of GT maps: {}".format(jpg_count, gt_count))

check_num_images()

with open("train_pair.txt", 'w+') as fout:
	for img in img_list:
		img_path = os.path.join(dataset_root, img)
		fout.write(f'{img_path}.jpg {img_path}.png\n')