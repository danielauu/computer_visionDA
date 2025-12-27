import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path_in = 'images\local_descriptors_task'
name_in1 = 'patch1.tif'
name_in2 = 'lookup.tif'

name_out = 'patch1.png'
path_out = 'correction/descriptors'
img1 = cv.imread(os.path.join(path_in, name_in1))
img1.astype('float32')

img2 = cv.imread(os.path.join(path_in, name_in2))
img2.astype('float32')
################################################### SIFT
# sift = cv.xfeatures2d.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# #feature matching
# bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

# matches = bf.match(descriptors_1, descriptors_2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
# cv.imwrite(os.path.join(path_out, 'sift'+name_out), img3)


################################################### SURF
# surf = cv.xfeatures2d.SURF_create()

# keypoints_1, descriptors_1 = surf.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = surf.detectAndCompute(img2,None)

# #feature matching
# bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

# matches = bf.match(descriptors_1, descriptors_2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
# cv.imwrite(os.path.join(path_out, 'surf'+name_out), img3)
################################################### ORB
orb = cv.ORB_create(nfeatures=1500)
keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

#feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
cv.imwrite(os.path.join(path_out, 'orb'+name_out), img3)