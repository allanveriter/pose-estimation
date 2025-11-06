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


def pixel2world(pixel, camera_index):
    N = CoordSys3D('N')
    pixel_homogenous = sp.Matrix([*pixel, 1])

    with open('cameraMatrix' + str(camera_index) + '.pkl', 'rb') as file:
        camera_matrix = sp.Matrix(pickle.load(file))

    line_direction_camera = camera_matrix.inv() * pixel_homogenous  # same as ray
    line_point_camera = sp.Matrix([0, 0, 0])  # A point on the line in camera coordinates

    with open('rvec' + str(camera_index) + '.pkl', 'rb') as file:
        rvec = sp.Matrix(pickle.load(file))


    with open('tvec' + str(camera_index) + '.pkl', 'rb') as file:
        tvec = sp.Matrix(pickle.load(file))

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
    print("sp matrix: " + str(line_point_object))
    # Define the line point and direction
    line_point = Point(*line_point_object)
    print("sp point: " + str(line_point))
    line_direction = Point(*[line_point_object[i] + line_direction_object[i] for i in range(3)])
    # Define the line in 3D space
    line = Line(line_point, line_direction)


    # Define the xz-plane (y = 0) with normal vector pointing in the y direction
    plane_point = Point(0, 0, 0)
    plane_normal = [0, 1, 0]
    plane = Plane(plane_point, plane_normal)

    # Find the intersection
    intersection = line.intersection(plane)
    point1_tuple = tuple(float(coord) for coord in intersection[0].evalf())

    if intersection:
        return point1_tuple
    else:
        print("No intersection.")
        return None
