'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_mapping(CHESSBOARD_SIZE, SQUARE_SIZE_MM, camera_index):

    # Prepare object points based on the real chessboard size
    object_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_MM / 1000.0  # Convert from mm to meters

    # Arrays to store object points and image points from all the images
    obj_points = []
    img_points = []

    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use camera with ID 0
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
        frame_small = cv2.resize(frame, (1500, 800))
        cv2.imshow('Camera Calibration', frame_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if ret:
                print("Captured an image with chessboard corners.")
                img_points.append(corners)
                obj_points.append(object_points)
                break

    cap.release()
    cv2.destroyAllWindows()
    def plot_dict(dict):
        #################################"

        # rename to prevent TypeError: unbound
        coord_map = dict


        x_2d = [coord[0] for coord in coord_map.keys()]
        y_2d = [coord[1] for coord in coord_map.keys()]
        z_3d = list(coord_map.values())

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_2d, y_2d, color='blue')

        # Annotate each point with its corresponding 3D coordinate
        for i, coord in enumerate(z_3d):
            plt.text(x_2d[i] + 0.1, y_2d[i] + 0.1, str(coord), fontsize=5)

        # Label the axes
        plt.xlabel('X 2D')
        plt.ylabel('Y 2D')
        plt.title('2D Plot with 3D Coordinates Annotated')
        plt.gca().invert_yaxis()

        # Display the plot
        plt.grid(True)
        plt.show()
        ##################################

    img_points = img_points[0]
    img_points = [tuple(sublist[0]) for sublist in img_points]
    object_points = [tuple(sublist) for sublist in object_points]
    object_points = [(x*1000, z, y*1000) for (x, y, z) in object_points] # meters to millimeters

    dictionary = {tuple(img_points[i]): tuple(object_points[i]) for i in range(len(img_points))}
    plot_dict(dictionary)
    print(dictionary)
    import json

    # The list variable we want to write to a file

    # Writing to a text file
    # Write to a binary file using pickle
    with open('dictionary.pkl', 'wb') as file:
        pickle.dump(dictionary, file)
    return len(dictionary)


