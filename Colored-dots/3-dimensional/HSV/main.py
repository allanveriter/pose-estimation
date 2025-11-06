'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

from HSV_calibration import camera_hsv_mask

# Example usage
camera_port = 2  # Change this if your camera is on a different port
camera_name = 3
camera_hsv_mask(camera_port, camera_name)