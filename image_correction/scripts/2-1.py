import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Grayscale, RGB, YCbCr, HSV, XYZ, Lab
path_in = 'images'

name_out = 'blue-HMRegistred.png'
path_out = 'correction/Histograms/HM'

# #name_out = 'P63Registred.png'
# #path_out = 'correction/Histograms/P63'

blue = cv.imread(os.path.join(path_in, name_out))
blue.astype('float32')


## Chequin the correct convertion
rgb = cv.cvtColor(blue, cv.COLOR_BGR2RGB)
color = ('y','c','r')
Lab = cv.cvtColor(blue, cv.COLOR_BGR2Lab)


print(rgb[0][0])
print(Lab[0][0])
pLab = Lab[0][0]
pRGB = rgb[0][0]

# # Get XYZ format of one pixel:
X = round(0.412453*pRGB[0]+0.357580*pRGB[1]+0.180423*pRGB[2])
Y = round(0.212671*pRGB[0]+0.715160*pRGB[1]+0.072169*pRGB[2])
Z = round(0.019334*pRGB[0]+0.119193*pRGB[1]+0.950227*pRGB[2])
XYZ = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
print(XYZ[0][0] == [X, Y, Z])

# Get Lab format of one pixel:
X = X/0.950456
Z = Z/1.088754

L = 116*(Y**(1/3)) - 16

def f(t):
    if (t>0.008856):
        return t**(1/3)
    else:
        return 7.787*t+16/116

delta = 128

a = 500*(f(X) - f(Y)) + delta
b = 200*(f(Y) - f(Z)) + delta
print(L,a,b)

# Select the small part of the image containing only nuclei and
# compute its Lab color model(target) - average L / a / b values
L,a,b = cv.split(Lab)
avgL = np.mean(L)
avga = np.mean(a)
avgb = np.mean(b)


# Compute delta Lab - difference image between the input and the target
def diff(pix):
    return np.sqrt((pix[0]-avgL)**2+(pix[1]-avga)**2+(pix[2]-avgb)**2)

name = 'HMRegistred.png'

img = cv.imread(os.path.join(path_in, name))
img.astype('float32')
Labimg = cv.cvtColor(img, cv.COLOR_BGR2Lab)

diff_image = np.zeros((Labimg.shape[0],Labimg.shape[1],1), np.float32)
for y in range(Labimg.shape[1]):
    for x in range(Labimg.shape[0]):
        diff_image[x][y] = diff(Labimg[x][y])

cv.imwrite('correction/Lab/HMRegistred.png', diff_image)
cv.imshow('diff', diff_image)
cv.waitKey(0)
cv.destroyAllWindows()

path_in = 'correction/Lab/HMRegistred.png'
img = cv.imread(path_in)
img.astype('float32')
print(img)
ret,thresh1 = cv.threshold(img,21,255,cv.THRESH_BINARY)
cv.imwrite('thresh_HMRegistered.png', thresh1)
