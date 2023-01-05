import json

import numpy as np
import cv2
import mediapipe as mp
import itertools
import math
from shapely.geometry import Polygon

class Detector(object):
    """
    Detector class with mediapipe
    """

    def __init__(self, cap, output_fname, is_file = False, show_res = True):
        """
        Initialization
        @param cap: cv2.VideoCapture object
        @param output_fname: name of output json file
        @param is_file: True if capture object belongs to video file, else False
        @param show_res: if True, result window will be displayed
        """
        self.show_res = show_res
        self.is_file = is_file
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.out_name = output_fname
        self.points = []
        self.counter = 0
        success, image = self.cap.read()
        # self.vid_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MP4V'), 15,
        #                                   (image.shape[1], image.shape[0]))

        self.NameVideo = 'Blinking Detect'
        # Initialize the mediapipe face mesh class.
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh_videos = self.mp_face_mesh.FaceMesh(static_image_mode = False, 
                                                           max_num_faces = 1,
                                                           min_detection_confidence = 0.5, 
                                                           min_tracking_confidence = 0.3)

        # Initialize the mediapipe drawing styles class.
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detectFacialLandmarks(self, image, face_mesh, display=True):
        '''
        This function performs facial landmarks detection on an image.
        Args:
            image:     The input image of person(s) whose facial landmarks needs to be detected.
            face_mesh: The face landmarks detection function required to perform the landmarks detection.
            display:   A boolean value that is if set to true the function displays the original input image,
                       and the output image with the face landmarks drawn and returns nothing.
        Returns:
            output_image: A copy of input image with face landmarks drawn.
            results:      The output of the facial landmarks detection on the input image.
        '''
        # Initialize the mediapipe drawing class.
        mp_drawing = mp.solutions.drawing_utils

        # Perform the facial landmarks detection on the image, after converting it into RGB format.
        results = face_mesh.process(image[:, :, ::-1])

        # Create a copy of the input image to draw facial landmarks.
        output_image = image[:, :, ::-1].copy()

        # Check if facial landmarks in the image are found.
        if results.multi_face_landmarks:

            # Iterate over the found faces.
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=output_image,
                                          landmark_list=face_landmarks,
                                          connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=self.mp_drawing_styles
                                          .get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(image=output_image,
                                          landmark_list=face_landmarks,
                                          connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=self.mp_drawing_styles
                                          .get_default_face_mesh_contours_style())

            # Return the output image in BGR format and results of facial landmarks detection.
            return np.ascontiguousarray(output_image[:, :, ::-1], dtype=np.uint8), results

    def isAndgleOpen(self, arr, ind, numbers):

        upper_point_ind = ind.index(numbers[0])
        lower_point_ind = ind.index(numbers[1])
        corner_ind = ind.index(numbers[2])
        sec_corner_ind = ind.index(numbers[3])
        
        arr = np.array(arr) # иначе не будет вычитания

        vector1 = arr[upper_point_ind] - arr[corner_ind]
        vector2 = arr[lower_point_ind] - arr[corner_ind]

        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector1, unit_vector2)

        eucl_dist = lambda ar1, ar2: math.sqrt(np.linalg.norm(ar1 - ar2))

        angle = np.arccos(dot_product) * 180 / np.pi  # angle in radian

        return angle, eucl_dist(arr[upper_point_ind], arr[lower_point_ind]) / eucl_dist(arr[corner_ind], arr[sec_corner_ind]), arr

 

    def getSize(self, image, face_landmarks, INDEXES):
        '''
        This function calculate the height and width of a face part utilizing its landmarks.
        Args:
            image:          The image of person(s) whose face part size is to be calculated.
            face_landmarks: The detected face landmarks of the person whose face part size is to 
                            be calculated.
            INDEXES:        The indexes of the face part landmarks, whose size is to be calculated.
        Returns:
            width:     The calculated width of the face part of the face whose landmarks were passed.
            height:    The calculated height of the face part of the face whose landmarks were passed.
            landmarks: An array of landmarks of the face part whose size is calculated.
        '''

        # Retrieve the height and width of the image.
        image_height, image_width, _ = image.shape

        # Convert the indexes of the landmarks of the face part into a list.
        INDEXES_LIST = list(itertools.chain(*INDEXES))

        # Initialize a list to store the landmarks of the face part.
        landmarks = []

        # Iterate over the indexes of the landmarks of the face part. 
        for INDEX in INDEXES_LIST:
            # Append the landmark into the list.
            landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                              int(face_landmarks.landmark[INDEX].y * image_height)])
            
        # ls_single_face=results.multi_face_landmarks[0].landmark
        #     for idx in ls_single_face:
        #         print(idx.x,idx.y,idx.z)

        # Calculate the width and height of the face part.
        _, _, width, height = cv2.boundingRect(np.array(landmarks))

        # Convert the list of landmarks of the face part into a numpy array.
        landmarks = np.array(landmarks)

        # Retrurn the calculated width height and the landmarks of the face part.
        return width, height, landmarks

    def isOpen(self, image, face_mesh_results, face_part, display=True):

        # Retrieve the height and width of the image.
        image_height, image_width, _ = image.shape

        thold = None
        # Check if the face part is mouth.
        if face_part == 'MOUTH':

            INDEXES = self.mp_face_mesh.FACEMESH_LIPS

        # Check if the face part is left eye.
        elif face_part == 'LEFT EYE':

            INDEXES = self.mp_face_mesh.FACEMESH_LEFT_EYE

        # Check if the face part is right eye.
        elif face_part == 'RIGHT EYE':

            # Get the indexes of the right eye.
            INDEXES = self.mp_face_mesh.FACEMESH_RIGHT_EYE
          
            
        else:
            return

        # Iterate over the found faces.
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

            # Get the height of the face part.
            _, height, _ = self.getSize(image, face_landmarks, INDEXES)

            # Get the height of the whole face.
            _, face_height, _ = self.getSize(image, face_landmarks, self.mp_face_mesh.FACEMESH_FACE_OVAL)

            INDEXES_LIST = list(itertools.chain(*INDEXES))
            landmarks = []
            for INDEX in INDEXES_LIST:
                landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                                  int(face_landmarks.landmark[INDEX].y * image_height)])

            width = 0
            factor = 0
            size = 0

            if face_part == 'RIGHT EYE':
                upper_point_ind = 159
                lower_point_ind = 145
                corner_ind = 33
                sec_corner_ind = 133
                numbers = [upper_point_ind, lower_point_ind, corner_ind, sec_corner_ind]
                # Xcoords = 
                width, factor, array = self.isAndgleOpen(landmarks, INDEXES_LIST, numbers)
                size = (height / face_height) * 100

            if  face_part == 'LEFT EYE':
                upper_point_ind = 386
                lower_point_ind = 374
                corner_ind = 263
                sec_corner_ind = 362
                numbers = [upper_point_ind, lower_point_ind, corner_ind, sec_corner_ind]
                width, factor, array = self.isAndgleOpen(landmarks, INDEXES_LIST, numbers)
                size = (height / face_height) * 100
                
            if face_part == 'MOUTH':
                upper_point_ind = 0
                lower_point_ind = 17
                corner_ind = 61
                sec_corner_ind = 91
                numbers = [upper_point_ind, lower_point_ind, corner_ind, sec_corner_ind]
                width, factor, array = self.isAndgleOpen(landmarks, INDEXES_LIST, numbers)
                size = (height / face_height) * 100

        return width, size, factor, array
    
    
    def getArea(self, image, face_part, face_mesh_results, display=True):
        # https://stackoverflow.com/questions/69167499/mediapipe-assign-the-landmarks-to-the-vertices
        
        shape = image.shape
        
        for face in face_mesh_results.multi_face_landmarks:
            coords = []
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y      
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                coords.append([relative_x, relative_y])
                
        xx = []
        yy = []
               
        if face_part == 'RIGHT_CHEEK':
            INDEX = [123, 50, 36, 203,206, 186, 57, 202, 210, 135, 138, 213, 147]
            
        elif face_part == 'LEFT_CHEEK':    
            INDEX = [266, 280, 352, 376,433, 367, 364, 430, 422, 287, 410, 426, 423]
            
        elif face_part == 'CHIN': 
            INDEX = [170, 211, 204, 106, 182, 83, 18, 313, 406, 335, 424, 431, 395, 369, 396, 175,171, 140]
         
        elif face_part == 'FOREHEAD': 
            INDEX = [ 225, 224, 223, 222, 221, 55, 8,
                      285, 441, 442, 443, 444, 445,
                     276, 300, 
                      301, 298, 333, 299,
                     337, 151, 108, 69, 104, 68, 71, 
                     46, 70]
            
        elif face_part == 'FACE_OVAL':
            INDEX = [10,338,297,332,284,
                          251,389,356,454,323,
                          361,288,397,365,379,
                          378,400,377,152,148,
                          176,149,150,136,172,
                          58,132,93,234,127,162,
                          21,54,103,67,109]
            
        
        data = np.array(coords)
        for el in data[INDEX]:
            xx.append(el[0])
            yy.append(el[1])
            
        pgon = Polygon(zip(xx, yy))
        Area = pgon.area

        return Area
    
    def calcArea():
        pass

    def run(self):
        cv2.namedWindow(self.NameVideo, cv2.WINDOW_NORMAL)


        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success == True:

                frame = cv2.flip(frame, 1)
                try:
                    _, face_mesh_results = self.detectFacialLandmarks(frame,
                                                                      self.face_mesh_videos,
                                                                      display=False)
                    cv2.resizeWindow(self.NameVideo, 1000, 800)
                    cv2.imshow(self.NameVideo, frame)


                    if face_mesh_results.multi_face_landmarks:

                        lhold, lsize, lfactor, left_eye_coords = self.isOpen(frame,
                                                                               face_mesh_results,
                                                                               'LEFT EYE',
                                                                               display=False)

                        rhold, rsize, rfactor, right_eye_coords  = self.isOpen(frame, 
                                                                                face_mesh_results, 
                                                                                'RIGHT EYE',
                                                                                   display=False)
                        
                        mhold, msize, mfactor, mouth_coords  = self.isOpen(frame, 
                                                                                face_mesh_results, 
                                                                                'MOUTH',
                                                                                   display=False)
                        
                        right_cheek_area = self.getArea(frame,
                                                        'RIGHT_CHEEK',
                                                        face_mesh_results,
                                                        display=False)
                        
                        left_cheek_area = self.getArea(frame,
                                                        'LEFT_CHEEK',
                                                        face_mesh_results,
                                                        display=False)
                        
                        chin_area = self.getArea(frame,
                                                        'CHIN',
                                                        face_mesh_results,
                                                        display=False)
                        
                        forehead_area = self.getArea(frame,
                                                        'FOREHEAD',
                                                        face_mesh_results,
                                                        display=False)
                        
                        face_area = self.getArea(frame, 
                                                  'FACE_OVAL', 
                                                  face_mesh_results, 
                                                  display=False)

                    if cv2.waitKey(1) & 0xFF == 27 and not self.is_file:
                        break

                    if cv2.getWindowProperty(self.NameVideo, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    self.points.append({'frame': self.counter,
                                        'fps': self.fps,
                                        'time': self.counter / self.fps,
                                        'left_angle': lhold,
                                        'right_angle': rhold,
                                        'mouth_angle': mhold,
                                        'left_size': lsize,
                                        'right_size': rsize,
                                        'mouth_size': msize,
                                        'rfactor': rfactor,
                                        'lfactor': lfactor,
                                        'mfactor': mfactor,
                                        'left_eye_coords': left_eye_coords.tolist(),
                                        'right_eye_coords': right_eye_coords.tolist(),
                                        'mouth_coords': mouth_coords.tolist(),
                                        'right_cheek_area': right_cheek_area,
                                        'left_cheek_area': left_cheek_area,
                                        'chin_area': chin_area,
                                        'forehead_area': forehead_area,
                                        'face_area': face_area,
                                        })
                    
                   
                            

                except:
                    pass
            else:
                break
            self.counter += 1



        self.cap.release()
        cv2.destroyAllWindows()

        with open(self.out_name, 'w') as f:
            # print(self.points)
            json.dump(self.points, f)
            print(self.points)
            

if __name__ == '__main__':
    cap = cv2.VideoCapture('../../VIDEOS/_FEDOR_/VIDEO/bc292f06-a9a2-11ec-b5ec-b469216ca443.mp4')
    det = Detector(cap,
                    'ha.json',
                    is_file = True)
    det.run()

    
    
    
