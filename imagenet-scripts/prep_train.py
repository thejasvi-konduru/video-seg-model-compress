"""
About : Create train directory for imagenet.
1) Extract ILSVRC2012_img_train.tar into <train> folder
2) Copy this python script into <train> folder
3) Go to <train> folder and execute this script

"""

import os
import sys

# Load all files

tars_path = "."
tar_files = []

for f in os.listdir(tars_path):
	fp = os.path.join(tars_path, f)
	if os.path.isfile(fp):
		if ".tar" in fp and not "ILSVRC2012" in fp:
			tar_files.append(f)

# Removing main tar file
for tar_file in tar_files:
	print("Processing ", tar_file)
	class_id = tar_file.replace(".tar","")

	# Remove the directory if exists
	if os.path.isdir(class_id):
		os.system("rm -rf %s"%(class_id))
	# Creating fresh directory
	os.system("mkdir %s"%(class_id))  
	os.system("tar -xvf %s -C %s"%(tar_file, class_id))
	os.system("rm %s"%(tar_file))
	
