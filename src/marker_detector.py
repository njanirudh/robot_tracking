import numpy as np
import cv2
import cv2.aruco as aruco

from utilities import *
from threaded_webcam import *

class MarkerFinder :
    """
    Class to find the Aruco Marker in the image and perform the
    required processing tasks to get the pose of the robot.
    """
    def __init__(self):
        # Width and Height of sheet in 'mm'
        self.sheet_width  = 952.5
        self.sheet_height = 685.8

        # Creating a 4*4 aruco marker
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        # Setting parameters for finding the Aruco Marker.
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeStep = 1
        self.parameters.adaptiveThreshConstant = 10
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    def set_sheet_corner_id(self,bl,tl,tr,br):
        """
        Setting the 'id' of the markers to be found in the sheet
        The four markers should be attached to the edge of the
        sheet.
        :param bl: Aruco marker id Bottom Left
        :param tl: Aruco marker id Top Left
        :param tr: Aruco marker id Top Right
        :param br: Aruco marker id Bottom Right
        :return: None
        """
        self.sheet_bl = bl
        self.sheet_tl = tl
        self.sheet_tr = tr
        self.sheet_br = br

    def calibrate_camera(self,t_vec , r_vec,camera_mat):
        """
        Setting the calibration parameters of the camera
        :param t_vec: Translation vector
        :param r_vec: Rotation vector
        :return:  None
        """
        self.camera_mat = camera_mat
        self.t_vec = t_vec
        self.r_vec = r_vec

    def set_vehicle_marker_id(self,v_id_front , v_id_back):
        """
        Setting the front and back marker id of the robot
        :param v_id_front: Robot front marker id
        :param v_id_back: Robot back marker id
        :return: None
        """

        self.vehicle_id_f = v_id_front
        self.vehicle_id_b = v_id_back

    def process_image(self,frame):
        """
        Input frame of the frame from the camera containing the
        sheet markers and the robot markers.
        :param frame: Input image frame
        :return: Image with the markers marked
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # lists of ids and the corners belonging to each marker
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict,
                                                              parameters=self.parameters)
        print(ids)
        if np.all(ids != None):

            try:
                # Detecting 4 corner markers of Sheet
                index_bl = np.where(ids == np.array(self.sheet_bl))[0][0]
                index_tl = np.where(ids == np.array(self.sheet_tl))[0][0]
                index_tr = np.where(ids == np.array(self.sheet_tr))[0][0]
                index_br = np.where(ids == np.array(self.sheet_br))[0][0]

                # Debug Marking!!
                aruco.drawDetectedMarkers(frame,[corners[index_bl]])
                aruco.drawDetectedMarkers(frame,[corners[index_tl]])
                aruco.drawDetectedMarkers(frame,[corners[index_tr]])
                aruco.drawDetectedMarkers(frame,[corners[index_br]])

                cv2.circle(frame, (corners[index_bl][0][0][0],corners[index_bl][0][0][1]), 3 , (0,255,0) ,-1)
                cv2.circle(frame, (corners[index_tl][0][0][0],corners[index_tl][0][0][1]), 3 , (0,255,0), -1)
                cv2.circle(frame, (corners[index_tr][0][0][0],corners[index_tr][0][0][1]), 3 , (0,255,0), -1)
                cv2.circle(frame, (corners[index_br][0][0][0],corners[index_br][0][0][1]), 3 , (0,255,0), -1)

                # Preprocessing image
                #cv2.undistort(frame,None,None)
                pnts = np.array([(corners[index_bl][0][0][0],corners[index_bl][0][0][1]),
                        (corners[index_tl][0][0][0],corners[index_tl][0][0][1]),
                        (corners[index_tr][0][0][0],corners[index_tr][0][0][1]),
                        (corners[index_br][0][0][0],corners[index_br][0][0][1])])
                warped = four_point_transform(frame,pnts)

                v_corners, v_ids, _ = aruco.detectMarkers(warped, self.aruco_dict,
                                                                      parameters=self.parameters)
                if np.all(v_ids != None):

                    # Detecting Car markers
                    index_vehicle_b = np.where(v_ids == np.array(self.vehicle_id_b))[0][0]
                    index_vehicle_f = np.where(v_ids == np.array(self.vehicle_id_f))[0][0]

                    aruco.drawDetectedMarkers(warped, [v_corners[index_vehicle_b]])
                    aruco.drawDetectedMarkers(warped, [v_corners[index_vehicle_f]])

                    # Calculating the Markers centre
                    self.f_marker_centre = get_marker_centre(v_corners[index_vehicle_b][0][0],
                                                        v_corners[index_vehicle_b][0][1],
                                                        v_corners[index_vehicle_b][0][2],
                                                        v_corners[index_vehicle_b][0][3])

                    self.b_marker_centre = get_marker_centre(v_corners[index_vehicle_f][0][0],
                                                        v_corners[index_vehicle_f][0][1],
                                                        v_corners[index_vehicle_f][0][2],
                                                        v_corners[index_vehicle_f][0][3])

                    # Removing errors using intrensic and extrensic matrix
                    warped = cv2.undistort(warped, self.camera_mat)
                    warped = cv2.projectPoints([pnts, self.b_marker_centre, self.f_marker_centre], self.r_vec,
                                               self.t_vec, self.camera_mat)

                    # Drawing the line from origin to robot markers
                    cv2.line(warped,(0,0),self.f_marker_centre,(255,0,255),3)
                    cv2.line(warped, (0,0),self.b_marker_centre, (255, 255, 255),3)

                    #print(warped.shape)

                    # Finding the conversion factor of pixel
                    # into 'mm' of actual sheet
                    width_ratio = (self.sheet_width/warped.shape[1])
                    height_ratio = (self.sheet_height/warped.shape[0])

                    # Conversion of robot aruco-markers coordinate in pixels
                    # into 'mm' using the calculated ratios
                    self.f_robot_cartisian = (int(width_ratio*self.f_marker_centre[0]),
                                            int(height_ratio*self.f_marker_centre[1]))

                    self.b_robot_cartisian = (int(width_ratio*self.b_marker_centre[0]),
                                              int(height_ratio*self.b_marker_centre[1]))
                    #print(self.f_robot_cartisian,self.b_robot_cartisian)
                    return warped

            except:
                return None

        return None

    def get_output(self):
        """
        Returns the output as dictionary of cartisian coordinates of the
        robot in both front and back.
        :return:
        """
        result = {}
        result["front"]  = self.f_robot_cartisian
        result["back"]   = self.b_robot_cartisian

        return result

if __name__ == "__main__":

    test_img = cv2.imread("/home/nj/HBRS/Studies/Sem-2/SEE/DataSets/Dataset/Main/Forward/IMG_20190413_225156.jpg")

    finder = MarkerFinder()
    finder.set_sheet_corner_id(0,1,2,3)
    finder.set_vehicle_marker_id(46,47)

    image = finder.process_image(test_img)
    #cv2.imwrite("/home/anirudh/Desktop/SEE/SEE-Project/Assignment 03/images/map.jpg", image)
    print(finder.get_output())

    # webcam = WebcamVideoStream().start()
    #
    # while True:
    #     frame = webcam.read()
    #
    #     nj = finder.process_image(frame)
    #     cv2.imshow("Frame", nj)
    #
    #     c = cv2.waitKey(30)
    #     if c == 27:
    #         break

