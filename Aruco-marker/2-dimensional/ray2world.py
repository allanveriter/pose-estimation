'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import numpy as np

def ray2world(line):
    """
    Finds the intersection of a 3D line with the plane y = k.
    
    Parameters:
    line : list of two 3-element arrays/lists
        line[0] = starting point [x0, y0, z0]
        line[1] = ending point [x1, y1, z1]
    k : float
        y-coordinate of the plane
    
    Returns:
    intersection : np.array of shape (3,)
        3D coordinates of the intersection point
        Returns None if the line is parallel to the plane
    """
    k = 5
    p0 = np.array(line[0][0], dtype=float)
    p1 = np.array(line[0][1], dtype=float)
    
    # Direction vector of the line
    d = p1 - p0
    
    # Check if line is parallel to the plane
    if d[1] == 0:
        return None  # No intersection or line lies in plane
    
    # Parameter t for line equation p = p0 + t * d
    t = (k - p0[1]) / d[1]
    
    # Compute intersection point
    intersection = p0 + t * d
    return tuple(list(intersection))


