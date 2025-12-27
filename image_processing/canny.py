import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resize image
def resize(img, scale):
    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# 1 Load Image #########################################################
img1 = cv2.imread("images\TCGA-18-5592-01Z-00-DX1.tif", cv2.IMREAD_COLOR)
original = img1.copy()
original = resize(original, 60)
cv2.imwrite('procesing-images/scale/scale1.png', original)
# To gray scale
img2 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", img2)
cv2.imwrite('procesing-images/gray/gray1.png', img2)
# 2 Pre-processing #####################################################
# Resize
#cv2.imshow("Grayscale", img1)

# Noise filter
filter = cv2.bilateralFilter(img2,20,20,20)
#filter = cv2.GaussianBlur(img1, (21, 21), 0)
cv2.imshow("filter", filter)
cv2.imwrite('procesing-images/Noise-filter/bilateral1.png', filter)
# 3 Binarization #######################################################
#th = cv2.adaptiveThreshold(Gaussian,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY_INV,11,2)
ret, th = cv2.threshold(filter, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite('procesing-images/Otsu-threshold/threshold1.png', th)
#ret, th = cv2.threshold(Gaussian, 105, 255, cv2.THRESH_BINARY)
th = abs(255 - th)
cv2.imshow("Threshold", th)
cv2.imwrite('procesing-images/Inversion/Inversion1.png', th)

# 4. Bubble segmentation ###############################################
kernel = np.ones((5,5),np.uint8)
#OPEN
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 1)
cv2.imshow("opening", opening)
cv2.imwrite('procesing-images/morph/open/open1.png', th)

#ELIPSE
kernel = np.ones((10,5),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_ELLIPSE, kernel, iterations=1)
cv2.imshow("erosion", closing)
cv2.imwrite('procesing-images/morph/elipse/elipse1.png', th)

canny = cv2.Canny(closing, 30, 100)
cv2.imshow("Canny", canny)
cv2.imwrite('procesing-images/canny/canny1.png', th)

######################################################################
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cells = 0
for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cells += 1
        cv2.drawContours(original, [i], -1, (0, 255, 0), 1),
        cv2.putText(original, str(cells), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imshow('img2', original)
cv2.imwrite('results/canny1.png', original)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(cells)