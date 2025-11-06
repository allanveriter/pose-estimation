'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

from mapping import get_mapping
from solvePnP import solvePnP

# x-axis is along the width of the chessboard
# y-axis is up
# z-axis is along the length of the chessboard
# origin is in one of the outside corners
# see chessboardOrientation.jpg for more information


CHESSBOARD_SIZE = (9, 14)  # number of inner corners per chessboard row and column
# a 10 on 15 chessboard would give (9,14) in terms of inner corners
SQUARE_SIZE_MM = 54.5 #in mm
camera_index = 1 # Camera from which opencv reads, 0 for internal, 1 -> ... for external
camera_name = 1 #reads from Cameramatrix and writes to tvec and rvec, name will be the SUFFIX

get_mapping(CHESSBOARD_SIZE, SQUARE_SIZE_MM,camera_index)
solvePnP(camera_name)

#tvec and rvec are the rotation and translation vector from global origin to camera
