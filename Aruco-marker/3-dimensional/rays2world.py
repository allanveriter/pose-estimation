'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import numpy as np

def rays2world(lines):
    # input is a list with 2 lines
    # each line is a tuple (p1, p2) where p1, p2 are 3D points (arrays of length 3)
    # output is the intersection point of the 2 lines, or close intersection point

    # mathematical process:
    # we take the direction vectors of the 2 lines
    # multiply it by a parameter l and parameter t (we are looking for these parameters)
    # add a fixed point laying on the line
    # now if we subtract these 2 new vectors from each other, we get a new vector connecting the 2 lines
    # we also know that vector must be perpendicular on the 2 lines
    # we now have enough information to solve the linear system
    # more information on rays2world.jpg

    # Get direction vectors of the two lines
    dir_vector1 = np.array(lines[0][1]) - np.array(lines[0][0])
    dir_vector2 = np.array(lines[1][1]) - np.array(lines[1][0])

    # Compute the cross product of the direction vectors
    perpendicular_vector = np.cross(dir_vector1, dir_vector2)

    # Build the coefficient matrix A
    A = np.array([
        [perpendicular_vector[i], -dir_vector1[i], dir_vector2[i]] for i in range(3)
    ], dtype='float64')

    # Get points on the two lines
    c1 = np.array(lines[0][0])
    c2 = np.array(lines[1][0])

    # Build the right-hand side vector b
    b = np.array([
        [c1[i] - c2[i]] for i in range(3)
    ], dtype='float64')

    # Solve the linear system
    f, t, l = np.linalg.solve(A, b).flatten()

    # Compute points on each line using parameters
    point_on_line1 = dir_vector1 * t + c1
    point_on_line2 = dir_vector2 * l + c2
    print("Error on corner: ", np.linalg.norm(point_on_line2-point_on_line1))


    # Midpoint of the shortest connecting segment
    midpoint = (point_on_line1 + point_on_line2) / 2.0

    return tuple(midpoint)





