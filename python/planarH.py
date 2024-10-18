import numpy as np
import cv2


def computeH(x1, x2):
	"""
	OUTPUT:
	H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
	equation
	"""
	#Q3.6
	#Compute the homography between two sets of points
	N = x1.shape[0]
	A = []

	for i in range(N):
		x_1, y_1 = x1[i]
		x_2, y_2 = x2[i]
		A.append([-x_2, -y_2, -1, 0, 0, 0, x_2 * x_1, y_2 * x_1, x_1])
		A.append([0, 0, 0, -x_2, -y_2, -1, x_2 * y_1, y_2 * y_1, y_1])

	A = np.array(A)

	U, S, Vt = np.linalg.svd(A)
	h = Vt[-1, :]
	H = h.reshape((3, 3))
	H2to1 = H / H[2, 2] # normalize??? idk if its needed but it matches cv2.homography better

	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	def normalize_points(points):
		#Compute the centroid of the points
		centroid = np.mean(points, axis=0)

		#Shift the origin of the points to the centroid
		points_centered = points - centroid
		scale = np.sqrt(2) / np.mean(np.linalg.norm(points_centered, axis=1))

		T = np.array([[scale, 0, -scale * centroid[0]],
					  [0, scale, -scale * centroid[1]],
					  [0, 0, 1]])

		points_h = np.hstack([points, np.ones((points.shape[0], 1))])
		points_normalized = (T @ points_h.T).T

		return points_normalized[:, :2], T
	

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	#Similarity transform 1
	x1_norm, T1 = normalize_points(x1)
	#Similarity transform 2
	x2_norm, T2 = normalize_points(x2)

	#Compute homography
	H_norm = computeH(x1_norm, x2_norm)


	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	H2to1 /= H2to1[2, 2]

	return H2to1




def computeH_ransac(x1, x2, k=1000, dist=1):
	"""
	OUTPUTS
	bestH2to1 - homography matrix with the most inliers found during RANSAC
	inliers - a vector of length N (len(matches)) with 1 at the those matches
		that are part of the consensus set, and 0 elsewhere.
	"""
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	N = x1.shape[0]
	bestH2to1 = None
	maxil = 0
	bestil = None

	for i in range(k):
		# sample 4 point pairs
		ind = np.random.choice(N, 4, replace=False)
		H = computeH(x1[ind], x2[ind])

		# projection pairs
		x2_h = np.hstack([x2, np.ones((N, 1))])
		x2_proj = (H @ x2_h.T).T
		x2_proj = x2_proj[:, :2] / x2_proj[:, 2][:, np.newaxis]

		# see if errors are greater than what we want
		errors = np.linalg.norm(x1 - x2_proj, axis=1)
		inliers = errors < dist
		numil = np.sum(inliers)

		if numil > maxil:
			maxil = numil
			bestH2to1 = H
			bestil = inliers

	return bestH2to1, bestil.astype(int)



def compositeH(H2to1, template, img):
	'''note: i did not inverse the image because I just adjusted the
	   		 the data outside of the function :('''
	#Create a composite image after warping the template image on top
	#of the image using the homography
	h, w = img.shape[:2]

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	warped_template = cv2.warpPerspective(template, H2to1, (w, h))

	#Create mask of same size as template
	mask = np.sum(warped_template, axis=2) > 0
	mask = mask.astype(np.uint8)

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	img_masked = img * (1 - mask[:, :, np.newaxis])
	composite_img = img_masked + warped_template
	
	return composite_img


