from __future__ import print_function
from sklearn.externals import joblib
from skimage import feature
import mahotas
import cv2
import imutils
import numpy as np

model = joblib.load('digits.cpickle')
im = input("Path to your image : ")


def center_extent(image, size):
    (W, H) = size

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width = W)

    else:
        image = imutils.resize(image, height = H)
    extent = np.zeros((H, W), dtype = "uint8")

    offsetX = (W - image.shape[1]) // 2
    offsetY = (H - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image
    CM = mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)
    return extent


def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([[1, skew, -0.5 * w * skew],[0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h),
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    image = imutils.resize(image, width = width)

    return image


def getHist(image):
    hist = feature.hog(image, orientations = 18, pixels_per_cell = (10,10), cells_per_block = (1,1), transform_sqrt = True)
    return hist


image = cv2.imread(im)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 30, 150)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key =lambda x: x[1])

for (c, _) in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    roi = gray[y:y + h, x:x + w]
    thresh = roi.copy()
    T = mahotas.thresholding.otsu(roi)
    thresh[thresh > T] = 255
    thresh = cv2.bitwise_not(thresh)
    thresh =deskew(thresh, 20)
    thresh = center_extent(thresh, (20, 20))
    hist = getHist(thresh)
    digit = model.predict([hist])[0]

    cv2.rectangle(image, (x, y), (x + w, y + h),(0, 255, 0), 1)
    cv2.putText(image, str(digit), (x - 10, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)