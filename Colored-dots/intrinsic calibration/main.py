'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

from get_cameraMatrix import get_cameraMatrix


CHESSBOARD_SIZE = (7, 10)  # number of inner corners per chessboard row and column
# a 10 on 15 chessboard would give (9,14) in terms of inner corners
SQUARE_SIZE_MM = 21.17 # size of a square in millimeters


camera_index = 1
camera_name = 1

# You will have to show your chessboard in front of the camera at different angles and positions
# Press 's' to capture an image.
# Capture at least 8 images for accurate results
# Press 'k' to stop capturing and calibrate.

get_cameraMatrix(CHESSBOARD_SIZE, SQUARE_SIZE_MM, camera_index, camera_name)

#writes into cameraMatrix+suffix.pkl