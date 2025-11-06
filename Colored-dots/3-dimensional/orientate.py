'''
Written, assembled, and tested by Allan Vériter
Based on code by Jan Lemeire
For questions about the code or bugs: allan.nathan.nicolas.veriter@vub.be

GitHub repository: https://github.com/allanveriter/pose-estimation
'''

#orientates the 4 corners of a drone as following:
#pink is the front of the drone, yellow the back
# [pink-left, pink-right, yellow-right, yellow-left]
def orientate(pink, yellow):
    if len(pink) < 2 or len(yellow) < 2:
        return None

    def find_center_of_tuples(tuples):
        # Unpacking tuples into separate lists of x, y, and z coordinates
        x_coords = [t[0] for t in tuples]
        y_coords = [t[1] for t in tuples]

        # Calculating the average of each coordinate
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        # Returning the center as a tuple
        return center_x, center_y

    # checks if 2 points are layed counterclockwise around a centrum or not
    def orientation(center, p1, p2):
        # Calculate the cross product of vectors (center -> p1) and (center -> p2)
        val = (p1[0] - center[0]) * (p2[1] - center[1]) - (p1[1] - center[1]) * (p2[0] - center[0])

        if val > 0:
            return True  # counterclockwise
        elif val < 0:
            return False  # clockwise

    center = find_center_of_tuples(pink + yellow)

    if not orientation(center, pink[0], pink[1]):
        pink = [pink[1], pink[0]]
    if not orientation(center, yellow[0], yellow[1]):
        yellow = [yellow[1], yellow[0]]

    pixel = pink + yellow
    print(pixel)
    return pixel
