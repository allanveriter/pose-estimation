'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import cv2
import numpy as np


# Function to save the HSV values to a file
def save_hsv_values(filename, lower, upper):
    with open(filename, 'w') as file:
        file.write(f"{lower}\n")
        file.write(f"{upper}\n\n")


# Callback function for the trackbars (does nothing but is needed for OpenCV)
def nothing(x):
    pass


# Function to open camera feeds and mask them with chosen HSV ranges
def camera_hsv_mask(camera_port, camera_name):

    # Open the camera
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

    # Create windows
    cv2.namedWindow("Camera Feed 1")
    cv2.namedWindow("Camera Feed 2")

    # Create two sets of trackbars for Pink and Yellow HSV ranges
    cv2.namedWindow("Pink Mask Sliders")
    cv2.namedWindow("Yellow Mask Sliders")

    # Pink sliders (HSV ranges)
    cv2.createTrackbar('H_low_pink', 'Pink Mask Sliders', 0, 179, nothing)
    cv2.createTrackbar('H_high_pink', 'Pink Mask Sliders', 179, 179, nothing)
    cv2.createTrackbar('S_low_pink', 'Pink Mask Sliders', 0, 255, nothing)
    cv2.createTrackbar('S_high_pink', 'Pink Mask Sliders', 255, 255, nothing)
    cv2.createTrackbar('V_low_pink', 'Pink Mask Sliders', 0, 255, nothing)
    cv2.createTrackbar('V_high_pink', 'Pink Mask Sliders', 255, 255, nothing)

    # Yellow sliders (HSV ranges)
    cv2.createTrackbar('H_low_yellow', 'Yellow Mask Sliders', 0, 179, nothing)
    cv2.createTrackbar('H_high_yellow', 'Yellow Mask Sliders', 179, 179, nothing)
    cv2.createTrackbar('S_low_yellow', 'Yellow Mask Sliders', 0, 255, nothing)
    cv2.createTrackbar('S_high_yellow', 'Yellow Mask Sliders', 255, 255, nothing)
    cv2.createTrackbar('V_low_yellow', 'Yellow Mask Sliders', 0, 255, nothing)
    cv2.createTrackbar('V_high_yellow', 'Yellow Mask Sliders', 255, 255, nothing)

    while True:
        # Read from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current positions of the sliders for Pink mask
        h_low_pink = cv2.getTrackbarPos('H_low_pink', 'Pink Mask Sliders')
        h_high_pink = cv2.getTrackbarPos('H_high_pink', 'Pink Mask Sliders')
        s_low_pink = cv2.getTrackbarPos('S_low_pink', 'Pink Mask Sliders')
        s_high_pink = cv2.getTrackbarPos('S_high_pink', 'Pink Mask Sliders')
        v_low_pink = cv2.getTrackbarPos('V_low_pink', 'Pink Mask Sliders')
        v_high_pink = cv2.getTrackbarPos('V_high_pink', 'Pink Mask Sliders')

        # Create the Pink mask
        lower_pink = np.array([h_low_pink, s_low_pink, v_low_pink])
        upper_pink = np.array([h_high_pink, s_high_pink, v_high_pink])
        pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)

        # Get current positions of the sliders for Yellow mask
        h_low_yellow = cv2.getTrackbarPos('H_low_yellow', 'Yellow Mask Sliders')
        h_high_yellow = cv2.getTrackbarPos('H_high_yellow', 'Yellow Mask Sliders')
        s_low_yellow = cv2.getTrackbarPos('S_low_yellow', 'Yellow Mask Sliders')
        s_high_yellow = cv2.getTrackbarPos('S_high_yellow', 'Yellow Mask Sliders')
        v_low_yellow = cv2.getTrackbarPos('V_low_yellow', 'Yellow Mask Sliders')
        v_high_yellow = cv2.getTrackbarPos('V_high_yellow', 'Yellow Mask Sliders')

        # Create the Yellow mask
        lower_yellow = np.array([h_low_yellow, s_low_yellow, v_low_yellow])
        upper_yellow = np.array([h_high_yellow, s_high_yellow, v_high_yellow])
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Apply the masks
        pink_result = cv2.bitwise_and(frame, frame, mask=pink_mask)
        yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

        # Show the masked camera feeds
        cv2.imshow('Camera Feed 1', pink_result)
        cv2.imshow('Camera Feed 2', yellow_result)
        # Break the loop and save HSV values if 's' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):

            print("type: "+ str(type(lower_pink)))
            save_hsv_values(f"pink{camera_name}.txt",lower_pink, upper_pink)
            save_hsv_values(f"yellow{camera_name}.txt", lower_yellow, upper_yellow)
            print(f"HSV values saved to {camera_name}")
        elif key == 27:  # ESC to exit
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()



