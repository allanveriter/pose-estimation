'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import os
import cv2
import numpy as np

def solvePnP(camera_name):
    import json
    import pickle
    # Reading from the text file
    with open('dictionary.pkl', 'rb') as file:
        dict = pickle.load(file)

    object_points = []
    image_points = []
    for key, value in dict.items():
        if True:
            object_points.append(value)
            image_points.append(key)


    # Define your 3D points on the table in real life
    object_points = np.array(object_points, dtype=np.float32)

    # Define the corresponding 2D pixel coordinates in the image
    image_points = np.array(image_points, dtype=np.float32)

    # Assuming a camera matrix (this could be different based on your camera)
    # fx and fy are focal lengths, and cx, cy are the optical centers.

    with open('cameraMatrix'+str(camera_name) + '.pkl', 'rb') as file:
        camera_matrix = pickle.load(file)
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Dictionary to hold results
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if success:
        # Project the 3D points to 2D using the estimated pose
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

        # Compute total error
        print("average Error: " + str(np.sum(np.square(image_points - projected_points.reshape(-1, 2))) / len(dict)))
    # The list variable we want to write to a file

    # Writing to a text file
    with open('rvec' + str(camera_name) + '.pkl', 'wb') as file:
        pickle.dump(rvec, file)
    with open('tvec' + str(camera_name) + '.pkl', 'wb') as file:
        pickle.dump(tvec, file)

    if os.path.exists('dictionary.pkl'):
        os.remove('dictionary.pkl')