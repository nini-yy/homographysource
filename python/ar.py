import numpy as np
import cv2

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

#Write script for Q4.1
def loadVid2(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return None

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return np.array(frames)

def crop_center(image, target_shape):
    h, w = image.shape[:2]
    th, tw = target_shape
    cy, cx = h // 2, w // 2
    sx = cx - (tw // 2)
    sy = cy - (th // 2)
    return image[sy:sy + th, sx:sx + tw]



bookframe2 = loadVid2('../data/book.mov')
#book_frames = loadVid('../data/book.mov')

arframe2 = loadVid2('../data/ar_source.mov')
#ar_frames = loadVid('../data/ar_source.mov')


h_book, w_book, _ = bookframe2[0].shape
#print(h_book, w_book)
#print(len(bookframe2))
aframes = arframe2[:w_book+1]


finalframes = []
for i in range(641):
    if i % 10 == 0:
        print(i)

    aone = arframe2[i]
    bone = bookframe2[i]

    #crop the center
    hcover, wcover, _ = cv_cover.shape
    caone = crop_center(aone, (hcover, wcover))

    matches, locs1, locs2 = matchPics(cv_cover, bone, 0.2, 0.7)

    locs1 = locs1[:, [1, 0]]  # Swap columns (y, x) -> (x, y)
    locs2 = locs2[:, [1, 0]]

    matched_points1 = locs1[matches[:, 0]]  # Points from cv_cover
    matched_points2 = locs2[matches[:, 1]]
    #print(len(matched_points1), len(matched_points2))
    if len(matched_points1) <= 3:
        print("less 10", len(matched_points1))
        continue
    if len(matched_points2) <= 3:
        print("less 10", len(matched_points2))
        continue

    bestH2to1, i = computeH_ransac(matched_points2, matched_points1, 10000, 10)

    h_desk, w_desk, _ = bone.shape
    h_cover, w_cover, _ = cv_cover.shape

    #resize to same size as cv_cover
    resized_caone_cover = cv2.resize(caone, (w_cover, h_cover))
    #warped_ca_cover = cv2.warpPerspective(resized_caone_cover, bestH2to1, (w_desk, h_desk))

    composite_img = compositeH(bestH2to1, resized_caone_cover, bone)
    finalframes.append(composite_img)


def frames_to_video(frames, output_path, fps=20.0):
    h, w, channels = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, w):
            print(f"Error: Frame {i} has different size. Skipping this frame.")
            continue  # Skip frames with incorrect dimensions
        out.write(frame)

    out.release()

    print("video saved")

# Example usage:
# Assuming `book_frames` is a list of frames (numpy arrays) already loaded
frames_to_video(finalframes, 'ar.avi')