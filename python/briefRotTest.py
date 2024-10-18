import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
import scipy


#Q3.5
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')


num_matches = []
for i in range(36):
	#Rotate Image
    angle = i * 10
    rotated = scipy.ndimage.rotate(img, angle)
	
	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(img, rotated, 0.2, 0.75)
    #print("i, match", i, len(matches))

	#Update histogram
    num_matches.append(len(matches))



#Display histogram
angles = [i * 10 for i in range(36)]
plt.figure(figsize=(10, 6))
plt.bar(angles, num_matches, width=9, align='center')  
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('Number of Matches vs. Image Rotation')

plt.ioff()
plt.savefig('rotations')
