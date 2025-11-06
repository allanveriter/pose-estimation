'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import cv2
import numpy as np
import pickle

def get_cameraMatrix(CHESSBOARD_SIZE, SQUARE_SIZE_MM, camera_index, camera_name):
    # Prepare object points based on the real chessboard size
    object_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_MM / 1000.0  # Convert from mm to meters

    # Arrays to store object points and image points from all the images
    obj_points = []
    img_points = []

    # Open camera
    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  Use camera with ID 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # Set the resolution you want

    print("Press 's' to capture an image.")
    print("Press 'k' to stop capturing and calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)

        # Resize the frame to 100x100 pixels
        frame_small = cv2.resize(frame, (350, 100))
        cv2.imshow('Camera Calibration', frame_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if ret:
                print("Captured an image with chessboard corners.")
                img_points.append(corners)
                obj_points.append(object_points)
        elif key == ord('k'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Perform camera calibration
    if len(obj_points) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        if ret:
            print("Calibration successful.")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)

            # Save the camera matrix and distortion coefficients to pickle files
            with open('cameraMatrix' + str(camera_name) + '.pkl', 'wb') as f:
                pickle.dump(mtx, f)
        else:
            print("Calibration failed.")
    else:
        print("No images captured. Calibration cannot be performed.")
