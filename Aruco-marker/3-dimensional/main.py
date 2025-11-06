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
import cv2.aruco as aruco
import numpy as np



# fucntion to graphically debug the lines and points seen by the cameras
def plot_3d_lines_and_quadrilateral(lines, points, length=10, xlim=(0, 1000), ylim=(0, 2000), zlim=(0, 2500)):
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




# Simulating a camera feed that returns a list of 4 values (between 0 and 10 or None)
class CameraFeed():
    def __init__(self, camera_id, camera_name, focus):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.focus = focus

        
        
        # --- Video capture setup ---
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)  # Change index if needed

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Set MJPEG for full HD 30FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if self.focus != -1:
            # Disable autofocus (some drivers use 0 = off, 1 = on)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Set manual focus value (range often 0–255)
            self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)

        #the camera matrix must be inversed in the init, saves a lot of time
        with open('CM/cameraMatrix' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.cameraMatrix = np.linalg.inv((pickle.load(file)))
        with open('vectors/rvec' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.rvec = np.array(pickle.load(file)).flatten()
        with open('vectors/tvec' + str(self.camera_name) + '.pkl', 'rb') as file:
            self.tvec = np.array(pickle.load(file)).flatten()

        

        ######## ARUCO #################
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        self.roi_bounds = None  # (x1, y1, x2, y2), None = full image

    
    
    def FindCorners(self, target_id=47):
        h, w = self.img.shape[:2]

        # Decide which region to search
        if self.roi_bounds is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = self.roi_bounds
            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        roi_img = self.img[y1:y2, x1:x2]

        # Detect markers in ROI
        corners, ids, _ = aruco.detectMarkers(roi_img, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == target_id:
                    # Get coordinates relative to full image
                    c = [(int(x + x1), int(y + y1)) for x, y in corners[i][0]]

                    # Update ROI: add 50px margin around detected marker
                    xs = [p[0] for p in c]
                    ys = [p[1] for p in c]
                    self.roi_bounds = (
                        max(min(xs) - 50, 0),
                        max(min(ys) - 50, 0),
                        min(max(xs) + 50, w),
                        min(max(ys) + 50, h)
                    )
                    return c

        # Marker not found in ROI → search whole image next time
        self.roi_bounds = None
        return 0
    

    def get_rays_from_camera(self):

        # Capture the current frame
        if self.focus != -1:
            self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        
        ret, self.img = self.cap.read()

        
        if not ret:
            print("Failed to grab frame.")
            return [None for i in range(4)]

        else:

            corners = self.FindCorners()

            
            if corners != 0:

            # Draw blueish points on the image (assuming corners contains 4 pixel coordinates)
                for point in corners:
                    cv2.circle(self.img, point, radius=12, color=(255, 100, 55), thickness=-1)  # Pink color (BGR format)

                
                if self.camera_name == 4:
                    window_name = "Annotated Image4"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name, self.img)  # Show the image
                    cv2.waitKey(1)
                if self.camera_name == 3:
                    window_name2 = "Annotated Image3"
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name2, self.img)  # Show the image
                    cv2.waitKey(1)
                if self.camera_name == 2:
                    window_name2 = "Annotated Image2"
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name2, self.img)  # Show the image
                    cv2.waitKey(1)
                if self.camera_name == 1:
                    window_name2 = "Annotated Image1"
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)  # Create window, allowing resize if needed
                    cv2.imshow(window_name2, self.img)  # Show the image
                    cv2.waitKey(1)
            else:
                if self.camera_name == 4:
                    window_name = "Annotated Image4"
                    # Make a copy so original image isn't modified
                    img_copy = self.img.copy()
                    cv2.putText(img_copy,"nothing found",
                        (10, 30),  # position (x, y) in pixels
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        1,  # font scale
                        (0, 0, 255),  # red in BGR
                        2,  # thickness
                        cv2.LINE_AA)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window, allowing resize
                    cv2.imshow(window_name, img_copy)  # Show the annotated image
                    cv2.waitKey(1)

                if self.camera_name == 3:
                    window_name2 = "Annotated Image3"
                    # Make a copy so original image isn't modified
                    img_copy2 = self.img.copy()
                    cv2.putText(
                    img_copy2,
                    "nothing found",
                    (10, 30),  # position (x, y) in pixels
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)  # Create window, allowing resize
                    cv2.imshow(window_name2, img_copy2)  # Show the annotated image
                    cv2.waitKey(1)

                if self.camera_name == 2:
                    window_name = "Annotated Image2"
                    # Make a copy so original image isn't modified
                    img_copy = self.img.copy()
                    cv2.putText(img_copy,"nothing found",
                        (10, 30),  # position (x, y) in pixels
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        1,  # font scale
                        (0, 0, 255),  # red in BGR
                        2,  # thickness
                        cv2.LINE_AA)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window, allowing resize
                    cv2.imshow(window_name, img_copy)  # Show the annotated image
                    cv2.waitKey(1)
                if self.camera_name == 1:
                    window_name = "Annotated Image1"
                    # Make a copy so original image isn't modified
                    img_copy = self.img.copy()
                    cv2.putText(img_copy,"nothing found",
                        (10, 30),  # position (x, y) in pixels
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        1,  # font scale
                        (0, 0, 255),  # red in BGR
                        2,  # thickness
                        cv2.LINE_AA)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window, allowing resize
                    cv2.imshow(window_name, img_copy)  # Show the annotated image
                    cv2.waitKey(1)

            self.img = None

        # als de camera 1 of meerder van de hoekpunten niet ziet, alles op None zetten
        if corners == 0:
            rays = [None for i in range(4)]
        else:
            rays = []
            for pixel in corners:
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
            #print("not enough valid lines")

        elif len(valid_lines) == 2:
            # Case 2: Two valid lines, return their intersection
            world.append(rays2world(valid_lines))

        else:
            print("more than 2 lines, not possibel fix it???")

    if not (None in world):
        pass
        # 3d debugger voor rechten en punten
        #plot_3d_lines_and_quadrilateral(lines_for_debug, world)
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
    camera_feeds = [CameraFeed(camera_id=8, camera_name=3, focus=10),
                    CameraFeed(camera_id=4, camera_name=4, focus=-1),
                    CameraFeed(camera_id=0, camera_name=2, focus=100 ),
                    CameraFeed(camera_id=12, camera_name=1, focus=-1),]

    # number of cameras
    num_cameras = len(camera_feeds)

    historyOfPositions = [] #list holding 3d coordinates of drone
    sum = 0
    for i in range(3000):         
        start_time = time.perf_counter()

        # Wait for all camera feeds to send their data
        data_lists = [[], [], [], []]
        for camera_id in range(num_cameras):

            list_of_rays = camera_feeds[camera_id].get_rays_from_camera() 
            
            #print("list of rays: ", list_of_rays)
            if not(None in list_of_rays):
                item = camera_feeds.pop(camera_id)
                camera_feeds.insert(0, item)
                for i in range(4):
                    data_lists[i].append(list_of_rays[i])
            if len(data_lists[0]) == 2:
                break


        # Now process the 4 lists and get the resulting position
        if len(data_lists[0]) == 2:
            result = process_camera_data(data_lists)
            print(f"Resulting position: {result}")
        else:
            print("Nothing found")
        #historyOfPositions.append(result)
        # Reset the barrier for the next iteration          
        
        end_time = time.perf_counter()

        print('FPS: ' + str(1/(end_time - start_time)))
        sum += 1/(end_time - start_time)
        print("=======================")
        #plotHistoryOfDrone(historyOfPositions)'''
    print("Average FPS: ", str(sum/3000))
if __name__ == "__main__":
    main()
