import numpy as np
import cv2
import skimage.io 
import skimage.color
import matplotlib.pyplot as plt

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

#Write script for Q3.9

def display_image(image, title="Image"):
    plt.figure(figsize=(10, 6))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # RGB image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title(title)
    plt.axis('off')
    plt.show()


matches, locs1, locs2 = matchPics(cv_cover, cv_desk, 0.23, 0.9)

locs1 = locs1[:, [1, 0]]  # swap (y, x) (x, y)
locs2 = locs2[:, [1, 0]]

matched_points1 = locs1[matches[:, 0]]  # make sure they match
matched_points2 = locs2[matches[:, 1]]
#print(len(matched_points1), len(matched_points2))

bestH2to1, i = computeH_ransac(matched_points2, matched_points1, 10000, 5)
#print(bestH2to1)
#print(i)

h_desk, w_desk, _ = cv_desk.shape
h_cover, w_cover, _ = cv_cover.shape
resized_hp_cover = cv2.resize(hp_cover, (w_cover, h_cover))
#warped_hp_cover = cv2.warpPerspective(resized_hp_cover, bestH2to1, (w_desk, h_desk))

#display_image(warped_hp_cover)

composite_img = compositeH(bestH2to1, resized_hp_cover, cv_desk)
#display_image(composite_img, "Composite Image")
cv2.imwrite('composite_image.jpg', composite_img)