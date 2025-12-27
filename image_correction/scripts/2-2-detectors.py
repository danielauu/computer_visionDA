import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path_in = 'images\local_descriptors_task'
name_in = 'patch1.tif'

name_out = 'patch1.png'
path_out = 'correction/detectors'

# #name_out = 'P63Registred.png'
# #path_out = 'correction/Histograms/P63'

img = cv.imread(os.path.join(path_in, name_in))
img.astype('float32')

################################################### Harris
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.14)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,255,0]
# cv.imwrite(os.path.join(path_out, 'harris'+name_out), img)


################################################### Fast
# # Initiate FAST object with default values
# fast = cv.FastFeatureDetector_create(threshold=35)
# # find and draw the keypoints
# kp = fast.detect(gray,None)
# img2 = cv.drawKeypoints(gray, kp, None, color=(255,0,0))
# # Print all default params
# print( "Threshold: {}".format(fast.getThreshold()) )
# print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
# print( "neighborhood: {}".format(fast.getType()) )
# print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# #cv.imwrite('correction/detectors/fast_true.png', img2)

# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img, None)
# print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# #cv.imwrite('correction/detectors/fast_false.png', img3)
# Hori = np.concatenate((img2, img3), axis=1)
# cv.imwrite(os.path.join(path_out, 'fast'+name_out), Hori)


################################################### SIFT
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite(os.path.join(path_out, 'sift'+name_out), img)



#cv.imshow('dst',img)
#if cv.waitKey(0) & 0xff == 27:
#    cv.destroyAllWindows()
