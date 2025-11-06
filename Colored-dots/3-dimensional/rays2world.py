'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

from sympy import Line3D, Point3D, Matrix, Line
import numpy as np


def rays2world(lines):
    #input is a list with 2 lines
    # output is the intersection point of the 2 lines, or close intersection point

    #mathematical process:
    #we take the direction vectors of the 2 lines
    #multiply it by a parameter l and parameter t (we are looking for these parameters)
    #add a fixed point laying on the line
    #now if we substract these 2 new vectors from each others, we get a new vector, connecting the 2 lines
    # we also know that  vector must be perpendicular on the 2 lines
    # we now have enough information to solve the linear system
    # more information on rays2world.jpg

    # Get direction vectors of the  two lines
    dir_vector1 = Matrix(lines[0].direction_ratio)
    dir_vector2 = Matrix(lines[1].direction_ratio)

    # Compute the cross product of the direction vectors
    perpendicular_vector = tuple(dir_vector1.cross(dir_vector2))

    A = np.array([
        [perpendicular_vector[i], -dir_vector1[i], dir_vector2[i]] for i in range(3)
    ], dtype='float64')

    c1 = tuple(lines[0].p1)
    c2 = tuple(lines[1].p1)

    b = np.array([
        [c1[i] - c2[i]] for i in range(3)
    ], dtype='float64')

    f, t, l = np.linalg.solve(A, b)

    point_on_line1 = tuple([dir_vector1[i] * t[0] + c1[i] for i in range(3)])
    point_on_line2 = tuple([dir_vector2[i] * l[0] + c2[i] for i in range(3)])

    midpoint = ((point_on_line1[i] + point_on_line2[i])/2 for i in range(3))
    return tuple(midpoint)


# Example usage:
if __name__ == "__main__":
    line1 = Line(Point3D(2.77, -0.93, 0), (-1.06, 1.56, 1.68))
    line2 = Line(Point3D(-4.15, -4.37, 0), (4, 5, 0.59))
    lines = [line1, line2]
    midpoint = rays2world(lines)
    print(midpoint)

