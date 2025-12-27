import cv2 as cv
import numpy as np
import os

name_out = 'HMRegistred.png'
path_out = 'correction/gamma'

def resize(img, scale):
    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv.LUT(src, table)

img = cv.imread('images\HMRegistred.png')
img = resize(img, 10)
gammaImg = gammaCorrection(img, 0.4)

Hori = np.concatenate((img, gammaImg), axis=1)
cv.imwrite(os.path.join(path_out, 'gamma'+name_out), Hori)

#cv.imshow('result', Hori)
#cv.waitKey(0)
#cv.destroyAllWindows()