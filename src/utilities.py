import cv2
import numpy as np
import math

#------------------Perspective Transformation ------------------
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# ---------------- Point transformation ------------------------
def get_conversion_ratio(pnt1,pnt2):
    marker_size = 22.5

    pnt_tuple_1 = np.array((pnt1[0],pnt1[1]))
    pnt_tuple_2 = np.array((pnt2[0],pnt2[1]))
    #print("<< ",np.linalg.norm(pnt_tuple_1 - pnt_tuple_2))
    return (np.linalg.norm(pnt_tuple_1 - pnt_tuple_2)/marker_size)

def get_robot_pose(pnt1,pnt2):
    centre = (int((pnt1[0] + pnt2[0]) / 2), int((pnt1[1] + pnt2[1]) / 2))
    return centre

def get_marker_centre(pnt1,pnt2,pnt3,pnt4):
    centre1 = (int((pnt1[0] + pnt2[0]) / 2), int((pnt1[1] + pnt2[1]) / 2))
    centre2 = (int((pnt3[0] + pnt4[0]) / 2), int((pnt3[1] + pnt4[1]) / 2))

    centre = (int((centre1[0] + centre2[0]) / 2), int((centre1[1] + centre2[1]) / 2))
    return centre

def get_points_distance(pnt1,pnt2):
    pnt_tuple_1 = np.array((pnt1[0], pnt1[1]))
    pnt_tuple_2 = np.array((pnt2[0], pnt2[1]))

    #print("<< ", np.linalg.norm(pnt_tuple_1 - pnt_tuple_2))
    return  np.linalg.norm(pnt_tuple_1 - pnt_tuple_2)

# ----------------------Geometric utilities -----------------------
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_vector(pt1,pt2):
    return (abs(int(pt1[0] - pt2[0])),abs(int(pt1[1] - pt2[1])))
# -------------------------------------------------------------------

