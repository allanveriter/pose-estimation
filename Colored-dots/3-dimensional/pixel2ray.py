'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''


import sympy as sp
import pickle
from sympy.geometry import Point, Line3D, Plane, Line
from sympy.vector import CoordSys3D
import time


def pixel2ray(pixel, camera_matrix, rvec, tvec):
    N = CoordSys3D('N')
    pixel_homogenous = sp.Matrix([*pixel, 1])


    line_direction_camera = camera_matrix.inv() * pixel_homogenous  # same as ray
    line_point_camera = sp.Matrix([0, 0, 0])  # A point on the line in camera coordinates

    theta = rvec.norm()  # Angle of rotation
    if theta != 0:
        u = rvec / theta  # Normalized axis of rotation
    else:
        u = sp.Matrix([1, 0, 0])  # Default to x-axis if zero rotation

    # Skew-symmetric matrix for the axis of rotation
    K = sp.Matrix([[0, -u[2], u[1]],
                   [u[2], 0, -u[0]],
                   [-u[1], u[0], 0]])
    # Compute the rotation matrix using Rodriguez's formula
    R = sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * K ** 2

    # Compute the line in the object coordinate system
    line_point_object = R.T * (line_point_camera - tvec)
    line_direction_object = R.T * line_direction_camera


    # Define the line point and direction
    #these 2 lines of code are the ones making the whole localization process slow, they take around 0.1 to 0.3 seconds each to run
    line_point = Point(*line_point_object, evaluate = False)
    line_direction = Point(*[line_point_object[i] + line_direction_object[i] for i in range(3)], evaluate = False)
    # Define the line in 3D space
    return Line(line_point, line_direction)

