import os
import cv2 as cv
from matplotlib import pyplot as plt

# Grayscale, RGB, YCbCr, HSV, XYZ, Lab
path_in = 'images'

name_out = 'HMRegistred.png'
path_out = 'correction/Histograms/HM'

# name_out = 'P63Registred.png'
# path_out = 'correction/Histograms/P63'

img = cv.imread(os.path.join(path_in, name_out))
img.astype('float32')

def get_hist(img, color, mod_col, ax, labels):
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        ax.plot(histr,color = col, label=labels[i])
        plt.xlim([0,256])
        ax.legend(loc="upper left")
    ax.set_title(mod_col)
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('No. of pixels')

fig, axs = plt.subplots(3, 2, figsize=(10, 12))
##########################################################
# Grayscale
# RGB[A] to Gray: Y ← 0.299⋅R + 0.587⋅G + 0.114⋅B
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
color = ('g')
mod_col = 'Gray'
labels = ['Gray']
get_hist(gray, color, mod_col, axs[0, 0], labels)


##########################################################
# RGB
color = ('r','g','b')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mod_col = 'RGB'
labels = ['R','G','B']
get_hist(rgb, color, mod_col, axs[1, 0], labels)


##########################################################

# YCbCr

# |Y | = 0.299(R-G) + G + 0.114(B-G)
# |Cb| = 0.5
# |Cr| =

YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
mod_col = 'YCbCr'
labels = ['Y','Cb','Cr']
get_hist(YCbCr, color, mod_col, axs[2, 0], labels)


##########################################################
# HSV

# V ← max(R,G,B)

#     ⎧ V − min(R,G,B)
# S ←   --------------  if V≠0
#     ⎨      V
#     ⎩     v=0          otherwise

#     ⎧ 60(G−B)/(V−min(R,G,B))      if V=R
# H ←   120+60(B−R)/(V−min(R,G,B))  if V=G
#     ⎨ 240+60(R−G)/(V−min(R,G,B))  if V=B
#     ⎩ 0                           if R=G=B

HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mod_col = 'HSV'
labels = ['H','S','V']
get_hist(HSV, color, mod_col, axs[0, 1], labels)

##########################################################
# XYZ

# |X|   |0.412453 0.357580 0.180423| |R|
# |Y| = |0.212671 0.715160 0.072169|*|G|
# |Z|   |0.019334 0.119193 0.950227| |B|

XYZ = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
mod_col = 'XYZ'
labels = ['X','Y','Z']
get_hist(XYZ, color, mod_col, axs[1, 1], labels)


##########################################################
# Lab

# X ← X/Xn, where Xn = 0.950456
# Z ← Z/Zn, where Zn = 1.088754

# L ←   116∗Y^{1/3} − 16 for Y>0.008856
#       903.3∗Y          for Y≤0.008856

# a←500(f(X)−f(Y))+delta
# b←200(f(Y)−f(Z))+delta
# where

# f(t)={t^{1/3}         for t>0.008856
#       7.787t+16/116   for t≤0.008856
# and

# delta={1280for 8-bit imagesfor floating-point images

Lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
mod_col = 'Lab'
labels = ['L','a','b']
get_hist(Lab, color, mod_col, axs[2, 1], labels)

####################################################################
fig.tight_layout()
plt.savefig(os.path.join(path_out, mod_col + '-' + name_out))
#plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()