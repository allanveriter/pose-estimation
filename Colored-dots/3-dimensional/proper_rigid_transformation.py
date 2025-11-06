'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''


import numpy as np
from scipy.spatial.transform import Rotation
import math

def compute_rigid_transformation(predicted):
    if None in predicted:
        return None
    # drone is looking in x-direction, y-direction is up, z-direction follows a right-handed cartesian coordinate system
    # pink is front of the drone
    #       [pink-left,          pink-right,        yellow-right,       yellow-left]
    rigid = [(56.25, 0, -56.25), (56.25, 0, 56.25), (-56.25, 0, 56.25), (-56.25, 0, -56.25)]
    """
    Compute the rigid transformation (rotation and translation) that aligns the rigid drone coordinates
    with the predicted coordinates. Returns the pitch, yaw, and roll instead of the rotation matrix.

    Parameters:
    predicted - List of tuples representing the predicted positions of the drone's helices.
    rigid - List of tuples representing the rigid positions of the drone's helices when centered at the origin.

    Returns:
    pitch - Rotation around the x-axis (in degrees)
    yaw - Rotation around the y-axis (in degrees)
    roll - Rotation around the z-axis (in degrees)
    t - (3, ) translation vector
    """
    # Convert lists to numpy arrays
    A = np.array(rigid)  # Known rigid positions - lokaal
    B = np.array(predicted)  # Predicted positions - observed


    # Step 1: Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Step 2: Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Step 3: Compute the covariance matrix
    H = A_centered.T @ B_centered
    H = np.array(H, dtype=np.float64)

    # Step 4: Compute SVD (Kabsch Algorithm)
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Compute translation
    t = centroid_B - R @ centroid_A

    # Step 7: Convert rotation matrix to Euler angles (pitch, yaw, roll)
    r = Rotation.from_matrix(R)
    pitch, yaw, roll = r.as_euler('xyz', degrees=True)  # 'xyz' corresponds to roll, pitch, yaw

    # Step 8: Apply the rigid transformation to the rigid points
    transformed_rigid = []
    for point in rigid:
        transformed_point = R @ point + t  # Rotate and translate each point
        transformed_rigid.append(transformed_point)

    # Step 9: Calculate the distances between transformed rigid points and predicted points
    total_error = 0
    for i in range(4):
        distance = math.dist(transformed_rigid[i], predicted[i])
        total_error += distance

    # Step 10: Print the total error in mm
    print(f"Total error: {total_error:.2f} mm")

    return t
