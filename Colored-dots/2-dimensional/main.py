'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

from pixel2world import pixel2world
import cv2
import numpy as np
import math

################ HSV masks ################
img1 = None
mask_threshold = None

def HSVRangesMask(img_HSV, rangesLower, rangesUpper):
    global mask_threshold
    if rangesLower[0] < rangesUpper[0]:  # Hue-component
        mask_threshold = cv2.inRange(img_HSV, rangesLower, rangesUpper)
    else:
        range_upper2 = (256, rangesUpper[1], rangesUpper[2])
        mask_threshold1 = cv2.inRange(img_HSV, rangesLower, range_upper2)
        range_lower2 = (0, rangesLower[1], rangesLower[2])
        mask_threshold2 = cv2.inRange(img_HSV, range_lower2, rangesUpper)
        mask_threshold = cv2.bitwise_or(mask_threshold1, mask_threshold2)
    return mask_threshold

def maskedImageByHSVRanges(img, rangesLower, rangesUpper, imgHSV=None):
    global mask_threshold
    if imgHSV is None:
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_threshold = HSVRangesMask(imgHSV, rangesLower, rangesUpper)
    return cv2.bitwise_and(img, img, mask=mask_threshold)

def applyHSVRanges(camera_index):
    global low_H, low_S, low_V, high_H, high_S, high_V
    rangesLower = (low_H, low_S, low_V)
    rangesUpper = (high_H, high_S, high_V)
    img_masked = maskedImageByHSVRanges(globals()[f"img{camera_index}"], rangesLower, rangesUpper)
    cv2.imshow(globals()[f"window_thresholded_image_{camera_index}"], img_masked)

def setLowH(val):
    global low_H
    low_H = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

def setHighH(val):
    global high_H
    high_H = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

def setLowS(val):
    global low_S
    low_S = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

def setHighS(val):
    global high_S
    high_S = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

def setLowV(val):
    global low_V
    low_V = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

def setHighV(val):
    global high_V
    high_V = int(val)
    applyHSVRanges(1)
    applyHSVRanges(2)

################ Contours ################
def generate_hexagon(center, radius):
    angle = np.linspace(0, 2 * np.pi, 7)  # 6 vertices + 1 to close the hexagon
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return np.int32(np.vstack((x, y)).T)

def showContours(img, img_hsv, rangesTuple, window_name, camera_index):
    contours, _ = contoursOfHSVRange(img_hsv, rangesTuple[0], rangesTuple[1])
    img_contour = img.copy()

    if contours:
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        areas = [w * h for x, y, w, h in bounding_boxes]
        max_area_index = np.argmax(areas)
        biggest_contour = contours[max_area_index]

        cv2.drawContours(img_contour, [biggest_contour], -1, (0, 255, 0), 2)

        epsilon = 0.05 * cv2.arcLength(biggest_contour, True)
        polygon = cv2.approxPolyDP(biggest_contour, epsilon, True)

        if len(polygon) >= 1:
            x, y, w, h = cv2.boundingRect(biggest_contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            hexagon_points = generate_hexagon(center, radius)
            cv2.polylines(img_contour, [hexagon_points], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img_contour, center, 5, (0, 0, 0), -1)

            text = f"(X: {int(pixel2world(center, camera_index)[0])} mm)"
            text_position = (center[0] + 10, center[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 4

            cv2.putText(img_contour, text, (text_position[0] - 2, text_position[1] + 2), font, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(img_contour, text, text_position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            text = f"(Z: {int(pixel2world(center, camera_index)[2])} mm)"
            text_position = (center[0] + 10, center[1] + 80)

            cv2.putText(img_contour, text, (text_position[0] - 2, text_position[1] + 2), font, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(img_contour, text, text_position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow(window_name, img_contour)
    if contours:
        if len(polygon) >= 1:
            return center

def contoursOfHSVRange(img_hsv, range_lower, range_upper):
    if range_lower[0] < range_upper[0]:  # Hue-component
        mask = cv2.inRange(img_hsv, range_lower, range_upper)
    else:
        range_upper2 = np.array([256, range_upper[1], range_upper[2]])
        mask1 = cv2.inRange(img_hsv, range_lower, range_upper2)
        range_lower2 = np.array([0, range_lower[1], range_lower[2]])
        mask2 = cv2.inRange(img_hsv, range_lower2, range_upper)
        mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

################ GUI ################
max_value = 255
max_value_H = 180
low_H = 0
low_S = 0
low_V = 0

high_H = max_value_H
high_S = max_value
high_V = max_value

window_thresholds = 'Thresholds'
window_thresholded_image_1 = 'Thresholded image 1'
window_contours_1 = 'Contours 1'

cv2.namedWindow(window_thresholded_image_1, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_contours_1, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_thresholds, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_thresholds, 300, 200)

cv2.createTrackbar('Low H', window_thresholds, low_H, max_value_H, setLowH)
cv2.createTrackbar('High H', window_thresholds, high_H, max_value_H, setHighH)
cv2.createTrackbar('Low S', window_thresholds, low_S, max_value, setLowS)
cv2.createTrackbar('High S', window_thresholds, high_S, max_value, setHighS)
cv2.createTrackbar('Low V', window_thresholds, low_V, max_value, setLowV)
cv2.createTrackbar('High V', window_thresholds, high_V, max_value, setHighV)

################ MAIN ################
if __name__ == "__main__":
    print('** Test thresholds on the HSV colors **')
    print('Define ranges for the HSV-values.')

    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera with ID 1

    if not cap1.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret1, img1 = cap1.read()

        if not ret1:
            print("Failed to grab frame.")
            break

        img_HSV1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        applyHSVRanges(1)
        center1 = showContours(img1, img_HSV1, ((low_H, low_S, low_V), (high_H, high_S, high_V)), window_contours_1, 1)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # Press 'q' or 'ESC' to exit
            break

    cap1.release()
    cv2.destroyAllWindows()
