from collections import deque
from typing import Sequence

import math
import aria.sdk as aria
import fastplotlib as fpl
import numpy as np
import cv2
import shapely as shp
from enum import Enum

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


from common import ctrl_c_handler, quit_keypress

from projectaria_tools.core.sensor_data import (ImageDataRecord,)



#
# Histogram
# ---------------------------------------------------------------------
class HistHS:

    def __init__(self):

        self._back_hist = None
        self._bins = 25

    def calc_hsv(self, back_hsv:np.ndarray):
        self._back_hist = cv2.calcHist([back_hsv], [0, 1], None, [12, 12], [0, 181, 0, 256])
        #cv2.normalize(self._back_hist, self._back_hist, 0, 255, cv2.NORM_MINMAX, -1)

    def back_project(self, img:np.ndarray)->np.ndarray:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([img_hsv], [0, 1], self._back_hist, [0, 180, 0, 256], 1)
        return bp

    def back_project_mask(self, img:np.ndarray, disk_kernel)->np.ndarray:
        bp = self.back_project(img)
        dst = cv2.filter2D(bp, -1, disk_kernel)
        #_,mask = cv2.threshold(bp, 70, 255, 0)
        return dst

#
# Measure of Saliency
# ----------------------------------------
class Saliency:
    def __init__(self):
        self.saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual.create()
        self.saliency_fine = cv2.saliency.StaticSaliencyFineGrained.create()
        self.success = False

    def calc(self, img:np.ndarray)->bool:
        (success1, self.spectral_map) = self.saliency_spectral.computeSaliency(img)
        if success1:
            self.spectral_map = (self.spectral_map * 255).astype("uint8")
        (success2, self.fine_map) = self.saliency_fine.computeSaliency(img)
        if success2:
            # Apply log transformation method
            c = 255 / np.log(1 + np.max(self.fine_map))
            log_image = c * (np.log(self.fine_map + 1))

            # Specify the data type so that
            # float value will be converted to int
            self.fine_map =  np.array(log_image, dtype = np.uint8)
            #self.fine_map = (self.fine_map * 255).astype("uint8")

        self.success = success1 and success2
        return self.success

    #
    # Flood fill from black frame pixels to get a mask of teh background.
    #
    def floodfill(self)->np.ndarray:
        threshMap = cv2.threshold(self.spectral_map, 10, 255, cv2.THRESH_BINARY)[1]
        cols, rows,_ = threshMap.shape
        for i in range(cols):
            if threshMap[0, i] == 0:
                cv2.floodFill(threshMap, None, (i, 0), 255)
            if threshMap[rows - 1, i] == 0:
                cv2.floodFill(threshMap, None, (i, rows - 1), 255)
        for i in range(rows):
            if threshMap[i, 0] == 0:
                cv2.floodFill(threshMap, None, (0, i), 255)
            if threshMap[i, cols - 1] == 0:
                cv2.floodFill(threshMap, None, (cols - 1, i), 255)
        return threshMap

    def display_spectral(self, thresh=0)->np.ndarray:
        if not self.success:
            return None
        if thresh>0:
            disp = cv2.hconcat([self.spectral_map, cv2.threshold(self.spectral_map, 10, 255, cv2.THRESH_BINARY)[1]])
            return disp
        else:
            return self.spectral_map

    def display_fine(self, thresh=0)->np.ndarray:
        if not self.success:
            return None
        if thresh > 0:
            disp = cv2.hconcat([self.fine_map, cv2.threshold(self.fine_map, 10, 255, cv2.THRESH_BINARY)[1]])
            return disp
        else:
            return self.fine_map

    def combined_mask(self, thresh=0)->np.ndarray:
        images = [self.spectral_map, self.fine_map]
        if self.success:
            spectral_thresh = cv2.threshold(self.spectral_map, 10, 255, cv2.THRESH_BINARY)[1]
            fine_thresh = cv2.threshold(self.fine_map, 10, 255, cv2.THRESH_BINARY)[1]
            combined_mask = cv2.bitwise_and(spectral_thresh, fine_thresh)
            mask_median = cv2.medianBlur(combined_mask, 5)
            images.append(spectral_thresh)
            images.append(fine_thresh)
            images.append(mask_median)
            return cv2.hconcat(images)


class hand_pose(Enum):
    GRABBING = 1
    MID = 2
    FULL_OPEN = 3
    
    
    
    
# -----------------------------------------------------------------------
class SingleCameraObserver:
    """
    Observer that shrinks the image, prints out its shape,
    and optionally sends the shrunk frame over UDP.
    """
    
    # ----------------------------------------------------------------
    def read_image(fname:str)->np.ndarray:
        img = cv2.imread(str)
        if img is None:
            print(f"SingleCameraObserver:read_image: Could not read ",fname,".")
        else:
            print(fname, " loaded successfully!")
        return img 
        
    # ---------------------------------------------------------------
    def __init__(self):
        self._udp_socket = None
        self._remote_ip = "127.0.0.1"
        self._remote_port = 8899
        self._shrink_factor = 0.25

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hand = mp.solutions.hands.Hands()
        
        self.icons={}       
        # self.icons[hand_pose.GRABBING] = self.read_image("~/Code/ARIA/Icons/Hand_Grab.png") 
        # self.icons[hand_pose.MID] = self.read_image("~/Code/ARIA/Icons/Hand_Mid.png") 
        # self.icons[hand_pose.FULL_OPEN] = self.read_image("~/Code/ARIA/Icons/Hand_Full_Open.png") 

    def set(self, udp_socket=None, remote_ip="127.0.0.1", remote_port=8899, shrink_factor=0.25):
        self._udp_socket = udp_socket
        self._remote_ip = remote_ip
        self._remote_port = remote_port
        self._shrink_factor = shrink_factor

    # ------------------------------------------------------------------------
    def draw_debug_traingle(self, ind, pink, thumb, p_rt, ind_rt, cnt, color, img:np.ndarray=None):
        # cv2.line(img, ind, pink, (255,0,0), 2)
        # cv2.line(img, thumb, pink, (255,0,0), 2)
        # cv2.line(img, ind, thumb, (255,0,0), 2)         
        # # scaling distance
        # cv2.line(img, p_rt, ind_rt, (0,0,255), 2)
        # area
        cv2.circle(img, cnt, 20, color, -1)    
    
        
    #
    # if the hand open - measured by the area defined by the finger tips
    # ------------------------------------------------------------------------
    def is_hand_open(self, index_tip, pinky_tip, thumb_tip, pinky_root, index_root, root, h:int=0, w:int=0, img:np.ndarray=None)->hand_pose: 
    
        ind = (int(index_tip.x * w), int(index_tip.y * h))
        pink = (int(pinky_tip.x * w), int(pinky_tip.y * h))
        thumb = (int(thumb_tip.x * w), int(thumb_tip.y * h))         
        p_rt = (int(pinky_root.x * w), int(pinky_root.y * h))
        ind_rt = (int(index_root.x * w), int(index_root.y * h))
        rt = (int(root.x * w), int(root.y * h))
        
        polygon = shp.Polygon((ind, pink, thumb))
        length_sq = (math.pow(p_rt[0]-ind_rt[0], 2) + math.pow(p_rt[1]-ind_rt[1], 2) + math.pow(rt[0]-ind_rt[0], 2) + math.pow(rt[1]-ind_rt[1], 2))/2
        area = polygon.area/length_sq  
        #        print("polygon area = ", round(polygon.area, 2), "area = ", round(area,2), "len_sq = ", round(length_sq, 2))       
        
        color = (255,255,255)
        hand_state = hand_pose.FULL_OPEN
        if area<0.5:
            if area>=0.1:
               color = (0,255,0)
               hand_state = hand_pose.MID
            else:
                color = (0,0,255)
                hand_state = hand_pose.GRABBING
                   
        center = polygon.centroid
        if h>0 and w>0:
            self.draw_debug_traingle(ind, pink, thumb, p_rt, ind_rt, (int(center.x), int(center.y)), color, img)
        
        return hand_state

    # ------------------------------------------------------------------------
    def draw_finger_tips(self, index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip, h:int=0, w:int=0, img:np.ndarray=None): 
        cv2.circle(img, (int(index_tip.x * w), int(index_tip.y * h)), 5, (0,255,0), 2)
        cv2.circle(img, (int(middle_tip.x * w), int(middle_tip.y * h)), 5, (0,255,0), 2)
        cv2.circle(img, (int(ring_tip.x * w), int(ring_tip.y * h)), 5, (0,255,0), 2)
        cv2.circle(img, (int(pinky_tip.x * w), int(pinky_tip.y * h)), 5, (0,255,0), 2)
        cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 5, (0,255,0), 2)
        
    def on_image_received(self, image: np.ndarray)->np.ndarray:
        shrunk_image = cv2.resize(image, None, fx=self._shrink_factor, fy=self._shrink_factor, interpolation=cv2.INTER_AREA)

        h, w, _ = shrunk_image.shape
            
        #
        # Mediapipe hand tracking
        #
        result = self._hand.process(shrunk_image)
        
        if result.multi_hand_landmarks:
            
           for hand_landmarks in result.multi_hand_landmarks:
               
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                thumb_tip = hand_landmarks.landmark[4]
                pinky_root = hand_landmarks.landmark[17]
                index_root = hand_landmarks.landmark[5]
                root = hand_landmarks.landmark[0]

                #self.draw_finger_tips(index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip, h, w, img=shrunk_image)            
                self.is_hand_open(index_tip, pinky_tip, thumb_tip, pinky_root, index_root, root, h, w, shrunk_image)
    
        #Stream result frame.
        # if self._udp_socket:
        #     success, encoded = cv2.imencode(".jpg", shrunk_image)
        #     if success:
        #         frame_jpeg = encoded.tobytes()
        #         try:
        #             self._udp_socket.sendto(frame_jpeg, (self._remote_ip, self._remote_port))
        #         except Exception as e:
        #             print(f"Error sending UDP packet: {e}")
        return shrunk_image

#
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
class AriaVisualizer:

    def __init__(self):
        self.plots = fpl.Figure(shape=(1, 2), names=[["frame", "detection"]], size=(1200, 800) )
        # fpl.GridPlot(shape=(1, 2), size=(1200, 800) )
            

        self.image_plot = {
            0: self.plots[0, 1].add_image(np.zeros((352, 352, 3), dtype="uint8"), vmin=0, vmax=255, ),
            aria.CameraId.Rgb: self.plots[0, 0].add_image(
                np.zeros((1408, 1408, 3), dtype="uint8"), vmin=0, vmax=255, ),
        }
        self.singleCameraObserver = SingleCameraObserver()
        self.processed_image = np.zeros((352, 352, 3), dtype="uint8")


    # On any frame
    # --------------------------------------------------------------------------------------
    def set_single_camera_observer(self, udp_socket=None, remote_ip="127.0.0.1", remote_port=8899, shrink_factor=0.25):
        self.singleCameraObserver.set(udp_socket, remote_ip, remote_port, shrink_factor)


    def render_loop(self):
        self.plots.show()
        with quit_keypress and ctrl_c_handler(self.stop):
            fpl.loop.run()

    def stop(self):
        self.plots.close()


class BaseStreamingClientObserver:
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        print("BaseStreamingClientObserver to stream a frame...")
        pass
    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        pass


#
# --------------------------------------------------------------------------------------
#
class AriaVisualizerStreamingClientObserver(BaseStreamingClientObserver):
    def __init__(self, visualizer: AriaVisualizer):
        self.visualizer = visualizer

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
                       
        if record.camera_id == 2:
            image = np.rot90(image, k=3)
            self.visualizer.image_plot[2].data = image

            #print("stream a frame...")
            out_image = self.visualizer.singleCameraObserver.on_image_received(image)
            self.visualizer.image_plot[0].data = out_image

                
    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        print(f"Streaming Client Failure: {reason}: {message}")
