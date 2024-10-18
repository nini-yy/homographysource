import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, sigma=0.15, ratio=0.65):
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	i1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	i2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	i1loc = corner_detection(i1gray, sigma)
	i2loc = corner_detection(i2gray, sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(i1gray, i1loc)
	desc2, locs2 = computeBrief(i2gray, i2loc)
	
	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)


	return matches, locs1, locs2
	
