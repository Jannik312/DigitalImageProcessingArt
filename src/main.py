import cv2
import matplotlib.pyplot as plt
import numpy as np

# read image
img = cv2.imread('images/Jannik CV.jpg')
img = cv2.imread('images/Roman Path.jpg')

# resize to max 512px in one direction
max_px = max(img.shape[0], img.shape[1])
scale_factor = max_px/512
height = int(img.shape[0]/scale_factor)
width = int(img.shape[1]/scale_factor)
img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)



# convert to gray scale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')


# find edges using canny
def auto_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges


# show edges with canny
plt.imshow(auto_canny(gray), cmap='gray')

# find edges using laplacian:
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = laplacian - laplacian.min()
laplacian = laplacian / laplacian.max() * 255
laplacian = laplacian.astype('int')
plt.imshow(laplacian, cmap='gray')
plt.imshow(auto_canny(laplacian), cmap='gray')



# Harris Corner detection:
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
