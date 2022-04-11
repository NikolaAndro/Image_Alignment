'''
This program is used to warm an image to another perspective such that objects in the image align with each other.

Running command: python3 register.py destination_image.png source_img.png
'''
#########################################
#                                       #
#       Programmer: Nikola Andric       #
#       Email: namdd@umsystem.edu       #
#       Last Edited: 4/9/2022           #
#                                       #
#########################################

import sys
import cv2
from selectors import DefaultSelector
import numpy as np

#getting the name of the correct image
correct_img_name = sys.argv[1]

#getting the name of the correct image
incorrect_img_name = sys.argv[2]


# taking the image in grayscale format
correct_img = cv2.imread("../images/"+correct_img_name, cv2.IMREAD_GRAYSCALE)
incorrect_img = cv2.imread("../images/"+incorrect_img_name, cv2.IMREAD_GRAYSCALE)

#track the features of the image
# use sift algorithm to detect the features on the images
sift_algorithm = cv2.SIFT_create()

#create keypoints and descriptors of the image
key_points_correct_image, descriptors_correct_image = sift_algorithm.detectAndCompute(correct_img, None)
key_points_incorrect_image, descriptors_incorrect_image = sift_algorithm.detectAndCompute(incorrect_img, None)

# draw the keypoints on the image
correct_img_keypoints = cv2.drawKeypoints(correct_img, key_points_correct_image, correct_img)
incorrect_img_keypoints = cv2.drawKeypoints(incorrect_img, key_points_incorrect_image, incorrect_img)


# Let's see what key points of the original image are present on the image of the video
# feature matching between video stream and the static image (usinig flann algorithm to match the features since it is faster than ORB match detector)
index_params = dict(algorithm = 0, trees = 5)
search_params = dict()

flann_algorithm = cv2.FlannBasedMatcher(index_params, search_params)

# find matches using flann algorithm. Returning k best matches for each keypoint. 
matches = flann_algorithm.knnMatch(descriptors_correct_image, descriptors_incorrect_image, k=2)

# considering oly good matches
good_matches = []

# iterate over both images m - original, n - gray_frame
for m, n in matches:
    # determining how good the match is by comparing the distances.
    # the smaller the distance the better
    if m.distance < 0.5*n.distance:
        good_matches.append(m)

# draw the matches between the keypoints
# Note: It stacks two images horizontally and draw lines from first image to second image showing best matches. 
# There is also cv2.drawMatchesKnn which draws all the k best matches. If k=2, it will draw two match-lines for each keypoint. 
img3 = cv2.drawMatches(correct_img, key_points_correct_image, incorrect_img, key_points_incorrect_image, good_matches, incorrect_img)

while True:
    # show the images and frames in real time
    cv2.imshow("Destination Image",correct_img)
    cv2.imshow("Source Image",incorrect_img)
    cv2.imshow("Destination Image Keypoints",correct_img_keypoints)
    cv2.imshow("Source Image Keypoints",incorrect_img_keypoints)
    cv2.imshow("Matches",img3)

    key = cv2.waitKey(1)

    #if we press escape, we break the loop
    if key == 27:
        break

cv2.destroyAllWindows()


# get the points from good matches dictionary
query_pts = np.float32([key_points_correct_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
train_pts = np.float32([key_points_incorrect_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# using RANSAC algorithm to find the homography with outliers being removed.
homography_matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)


# Warp source image to destination based on homography
im_out = cv2.warpPerspective(incorrect_img, homography_matrix, (correct_img.shape[1],correct_img.shape[0]))

while True:

    # Display images
    cv2.imshow("Destination Image", correct_img)
    cv2.imshow("Source Image", incorrect_img)
    cv2.imshow("Warped Source Image", im_out)

    key = cv2.waitKey(1)

    #if we press escape, we break the loop
    if key == 27:
        break

cv2.destroyAllWindows()
