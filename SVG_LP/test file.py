import os
import numpy as np
import imageio
import cv2
data_root = "processed_video"
root_dir = data_root
data_dir = '%s/processed_data/train' % root_dir
ordered = False
dirs = []
for d1 in os.listdir(data_dir):
    for d2 in os.listdir('%s/%s' % (data_dir, d1)):
        dirs.append('%s/%s/%s' % (data_dir, d1, d2))
d = dirs[np.random.randint(len(dirs))]
print(d)

image_seq = []
fname = '%s/%d.png' % (d, 0)
print(fname)
print(cv2.imread(fname).shape)
img = cv2.imread(fname)
img = cv2.resize(img, (64, 64))
img = img.reshape(1, 64, 64, 3)
print(img.shape)

print(d)