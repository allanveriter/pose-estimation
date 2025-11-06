'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''



import threading
import random
import time
import cv2
import numpy as np
import os
from datetime import datetime
from pixel2ray import pixel2ray
from rays2world import rays2world
from proper_rigid_transformation import compute_rigid_transformation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from sympy import Point3D, Line3D, symbols, Matrix
import sympy as sp
import pickle
from orientate import orientate


# fucntion to graphically debug the lines and points seen by the cameras
def plot_3d_lines_and_quadrilateral(lines, points, length=10, xlim=(0, 400), ylim=(0, 100), zlim=(200, 600)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a symbol for parameter t
    t = symbols('t')

    # Plot the lines
    for line in lines:
        # Extract a point and direction vector from the line
        p = line.p1
        d = line.direction_ratio

        # Create the line equation in parametric form
        x = p[0] + t * d[0]
        y = p[1] + t * d[1]
        z = p[2] + t * d[2]

        # Substitute parameter values to get end points of the line segment for plotting
        t_values = [0, 2000]
        x_values = [x.subs(t, val).evalf() for val in t_values]
        y_values = [y.subs(t, val).evalf() for val in t_values]
        z_values = [z.subs(t, val).evalf() for val in t_values]

        ax.plot(x_values, y_values, z_values)

    # Plot and fill the quadrilateral
    if len(points) == 4:
        # Extract coordinates from the points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        # Create a list of vertices (points) for the quadrilateral
        vertices = [list(zip(xs, ys, zs))]
        poly3d = Poly3DCollection(vertices, alpha=.25, linewidths=1, edgecolors='r', facecolors='cyan')
        ax.add_collection3d(poly3d)

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.show()


# function to read HSV values from txt files
def read_hsv_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Expecting lines in the format: low_H, low_S, low_V, high_H, high_S, high_V
        low_H, low_S, low_V = (int(i) for i in lines[0][1:-2].split())
        high_H, high_S, high_V = (int(i) for i in lines[1][1:-2].split())
    return (low_H, low_S, low_V), (high_H, high_S, high_V)

def maskOfHSVRange(img_hsv, range_lower, range_upper):
    if range_lower[0] < range_upper[0]:
        mask = cv2.inRange(img_hsv, range_lower, range_upper)
    else:
        range_upper2 = np.array([256, range_upper[1], range_upper[2]])
        mask1 = cv2.inRange(img_hsv, range_lower, range_upper2)
        range_lower2 = np.array([0, range_lower[1], range_lower[2]])
        mask2 = cv2.inRange(img_hsv, range_lower2, range_upper)
        mask = cv2.bitwise_or(mask1, mask2)
    return mask

# Simulating a camera feed that returns a list of 4 values (between 0 and 10 or None)
class CameraFeed():
    def __init__(self, camera_id, camera_name):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name

        # Read HSV values from text files
        self.ranges_pink_lower, self.ranges_pink_upper = read_hsv_from_file('HSV/pink' + str(camera_name) + '.txt')
        self.ranges_yellow_lower, self.ranges_yellow_upper = read_hsv_from_file(
            'HSV/yellow' + str(camera_name) + '.txt')
        # Initialize the webcam
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        with open('CM/cameraMatrix' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.cameraMatrix = sp.Matrix(pickle.load(file))
        with open('vectors/rvec' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.rvec = sp.Matrix(pickle.load(file))
        with open('vectors/tvec' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.tvec = sp.Matrix(pickle.load(file))

        # Capture the initial frame and use it as the reference image
        # the first captured frames might be darker or lighter, let the camera adapt
        j = 0
        ret = None
        while j < 20:
            j += 1
            ret, self.start_img = self.cap.read()

        if ret:
            # Save the image as a JPG file
            cv2.imwrite('start_img' + str(camera_name) + '.jpg', self.start_img)
            print("Image saved successfully!")

        else:
            print("Failed to capture image")
            self.cap.release()
            cv2.destroyAllWindows()
            exit()

        # Convert the initial image to grayscale and apply Gaussian blur
        self.start_img = cv2.cvtColor(self.start_img, cv2.COLOR_BGR2GRAY)
        self.start_img = cv2.GaussianBlur(self.start_img, (21, 21), 0)

        ################ HSV masks ################
        self.img = None
        self.mask_threshold1 = None
        self.mask_threshold2 = None


    def maskOfContrast(self):
        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Blur the grayscale frame to reduce noise and detail
        gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)

        # Compute the absolute difference between the initial image and the current frame
        diff = cv2.absdiff(self.start_img, gray_frame)

        # Threshold the difference to get a binary image (highlighting the regions of interest)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in holes, making contours more solid
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the largest contour
        large_contours = []

        # Loop through the contours to find the largest one
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Only consider contours with at least 300 pixels
                large_contours.append(contour)
            # Create an empty mask the same size as the frame

        mask = np.zeros_like(thresh)

        # Draw the original contours onto the mask
        cv2.drawContours(mask, large_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Enlarge the contours by dilating the mask
        kernel = np.ones((21, 21), np.uint8)  # Adjust kernel size to expand by 10 pixels
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find the new, enlarged contours
        large_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return mask

    def contoursOf2Masks(self, mask1, mask2):
        new_mask = cv2.bitwise_and(mask2, mask2, mask=mask1)
        contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def showBiggestContours(self):
        mask1 = self.maskOfContrast()
        mask_pink = maskOfHSVRange(self.img_HSV, self.ranges_pink_lower, self.ranges_pink_upper)
        mask_yellow = maskOfHSVRange(self.img_HSV, self.ranges_yellow_lower, self.ranges_yellow_upper)
        contours_pink = self.contoursOf2Masks(mask1, mask_pink)
        contours_yellow = self.contoursOf2Masks(mask1, mask_yellow)

        def center_biggest_contours(contours):
            if contours:
                bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
                areas = [w * h for x, y, w, h in bounding_boxes]

                top_areas_idx = np.argsort(areas)[-2:]  # Get indices of the two largest areas
                pixel = []
                for idx in top_areas_idx:
                    x, y, w, h = bounding_boxes[idx]

                    center = (x + w // 2, y + h // 2)
                    pixel.append(center)
                return pixel
            return []

        pink1 = center_biggest_contours(contours_pink)
        yellow1 = center_biggest_contours(contours_yellow)
        for i in range(2 - len(pink1)):
            pink1.append(None)
        for i in range(2 - len(yellow1)):
            yellow1.append(None)
        return pink1, yellow1
    # compares actual image with initial image to know where to look for object

    def get_rays_from_camera(self):
        # Capture the current frame
        ret, self.img = self.cap.read()
        if not ret:
            print("Failed to grab frame.")
            return [None for i in range(4)]

        else:
            # Convert the image to HSV (if needed later in your process)
            self.img_HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

            # Assuming pink1 and yellow1 are obtained from self.showBiggestContours()
            pink1, yellow1 = self.showBiggestContours()

            if not(None in pink1 or None in yellow1):
            # Draw pink points on the image (assuming pink1 contains 2 pixel coordinates)
                for point in pink1:
                    cv2.circle(self.img, point, radius=5, color=(255, 0, 255), thickness=-1)  # Pink color (BGR format)

                    # Draw yellow points on the image (assuming yellow1 contains 2 pixel coordinates)
                for point in yellow1:
                    cv2.circle(self.img, point, radius=5, color=(0, 255, 255), thickness=-1)  # Yellow color (BGR format)

                if self.camera_name == 1:
                    window_name = "Annotated Image"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name, self.img)  # Show the image
                    cv2.waitKey(1)
                if self.camera_name == 3:
                    window_name2 = "Annotated Image2"
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name2, self.img)  # Show the image
                    cv2.waitKey(1)
            self.img = None

        # als de camera 1 of meerder van de hoekpunten niet ziet, alles op None zetten
        if None in pink1 or None in yellow1:
            rays = [None for i in range(4)]
        else:
            orientated_pixels = orientate(pink1, yellow1)
            rays = []
            for pixel in orientated_pixels:
                rays.append(pixel2ray(pixel, self.cameraMatrix, self.rvec, self.tvec))

        return rays


# Example function to process 4 lists and return a position
def process_camera_data(list_of_lists_of_lines):
    # list holding the 4 points of the drone
    world = []

    # list holding lines to plot later on
    lines_for_debug = []

    for i, lines in enumerate(list_of_lists_of_lines):
        valid_lines = [line for line in lines if line is not None]
        lines_for_debug += valid_lines
        # Step 2: Check the number of valid lines
        if len(valid_lines) <= 1:
            # Case 1: Only one or no valid line, return None
            world.append(None)

        elif len(valid_lines) == 2:
            # Case 2: Two valid lines, return their intersection
            world.append(rays2world(valid_lines))

        else:
            # Case 3: Three or more valid lines, calculate intersection of every pair
            intersections = []
            for i in range(len(valid_lines)):
                for j in range(i + 1, len(valid_lines)):
                    intersection = rays2world([valid_lines[i], valid_lines[j]])
                    if intersection is not None:
                        intersections.append(intersection)

            # If there are no valid intersections, return None
            if not intersections:
                return None

            # Step 4: Calculate the average of intersection points
            avg_x = sum(p[0] for p in intersections) / len(intersections)
            avg_y = sum(p[1] for p in intersections) / len(intersections)
            avg_z = sum(p[2] for p in intersections) / len(intersections)

            world.append((avg_x, avg_y, avg_z))

    if not (None in world):
        pass
        # 3d debugger voor rechten en punten
        # plot_3d_lines_and_quadrilateral(lines_for_debug, world)
    return compute_rigid_transformation(world)

def plotHistoryOfDrone(list_of_3d_coordinates):
    # Filter out None values
    filtered_coordinates = [coord for coord in list_of_3d_coordinates if coord is not None]

    # Unpacking the filtered coordinates into X, Y, Z lists
    x_vals = [x for x, y, z in filtered_coordinates]
    y_vals = [y for x, y, z in filtered_coordinates]
    z_vals = [z for x, y, z in filtered_coordinates]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x_vals, y_vals, z_vals)

    # Adding labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show plot
    plt.show()

def main():
    # Start multiple camera feeds
    camera_feeds = [CameraFeed(camera_id=1, camera_name=1),
                    CameraFeed(camera_id=2, camera_name=3), ]

    # number of cameras
    num_cameras = len(camera_feeds)

    historyOfPositions = [] #list holding 3d coordinates of drone

    while True:
        start_time = time.perf_counter()
        # Wait for all camera feeds to send their data
        data_lists = [[], [], [], []]
        for camera_id in range(num_cameras):
            list_of_rays = camera_feeds[camera_id].get_rays_from_camera()
            for i in range(4):
                data_lists[i].append(list_of_rays[i])

        # Now process the 4 lists and get the resulting position
        result = process_camera_data(data_lists)
        print(f"Resulting position: {result}")
        historyOfPositions.append(result)
        # Reset the barrier for the next iteration
        end_time = time.perf_counter()
        print('FPS: ' + str(1 / (end_time - start_time)))
        plotHistoryOfDrone(historyOfPositions)

if __name__ == "__main__":
    main()
