import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def save(im1, im2, style):
    Hori = np.concatenate((im1, im2), axis=1)
    cv.imwrite(os.path.join(path_out, style+name_out), Hori)

# Resize image
def resize(img, scale):
    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

path_in = 'images'

name_out = 'HMRegistred.png'
path_out = 'correction/equalization/HM'

#name_out = 'P63Registred.png'
#path_out = 'correction/Histograms/P63'

img = cv.imread(os.path.join(path_in, name_out))
img = resize(img, 10)
cv.imshow('img', img)
img.astype('float32')
################################################### Grayscale
# RGB[A] to Gray: Y ← 0.299⋅R + 0.587⋅G + 0.114⋅B

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#eqgray = cv.equalizeHist(gray)
#save(gray, eqgray, 'Gray')

################################################### RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

R, G, B = cv.split(rgb)
eqR = cv.equalizeHist(R)
eqG = cv.equalizeHist(G)
eqB = cv.equalizeHist(B)
eqRGB = cv.merge((eqR, eqG, eqB))
save(rgb, eqRGB, 'RGB')

################################################### YCbCr
YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

Y, Cb, Cr = cv.split(YCbCr)
eqY = cv.equalizeHist(Y)
eqCb = cv.equalizeHist(Cb)
eqCr = cv.equalizeHist(Cr)
eqYCbCr = cv.merge((eqY, eqCb, eqCr))
save(YCbCr, eqYCbCr, 'YCbCr')

################################################### HSV
HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

H, S, V = cv.split(HSV)
eqH = cv.equalizeHist(H)
eqS = cv.equalizeHist(S)
eqV = cv.equalizeHist(V)
eqHSV = cv.merge((eqH, eqS, eqV))
save(HSV, eqHSV, 'HSV')

################################################### XYZ
XYZ = cv.cvtColor(img, cv.COLOR_BGR2XYZ)

X, Y, Z = cv.split(XYZ)
eqX = cv.equalizeHist(X)
eqY = cv.equalizeHist(Y)
eqZ = cv.equalizeHist(Z)
eqXYZ = cv.merge((eqX, eqY, eqZ))
save(XYZ, eqXYZ, 'XYZ')

################################################### Lab
Lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)

L, a, b = cv.split(Lab)
eqL = cv.equalizeHist(L)
eqa = cv.equalizeHist(a)
eqb = cv.equalizeHist(b)
eqLab = cv.merge((eqL, eqa, eqb))
save(Lab, eqLab, 'Lab')
#cv.waitKey(0)
#cv.destroyAllWindows()