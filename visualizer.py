from collections import deque
from typing import Sequence

import math
import aria.sdk as aria
import fastplotlib as fpl
import numpy as np
from ultralytics import YOLO
import cv2
import shapely as shp
from enum import Enum
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


from common import ctrl_c_handler, quit_keypress

from projectaria_tools.core.sensor_data import (ImageDataRecord,)



#
# Histogram
# ===============================================================================================
class HistHS:

    def __init__(self):

        self._back_hist = None
        self._bins = 25

    def calc_hsv(self, back_hsv:np.ndarray):
        self._byolo_modelack_hist = cv2.calcHist([back_hsv], [0, 1], None, [12, 12], [0, 181, 0, 256])
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
# ===============================================================================================
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
    
    
#
# ===============================================================================================
class HistHS:

    def __init__(self):

        self._back_hist = None
        self._bins = 25

    def calc_hsv(self, back_hsv:np.ndarray):
        self._byolo_modelack_hist = cv2.calcHist([back_hsv], [0, 1], None, [12, 12], [0, 181, 0, 256])
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
    
# a detected device (out of tv, laptop, mouse, remote, keyboard, cell phone, microwave, book, clock)
# ===============================================================================================
class Device:

    def __init__(self):
        self._name = 'laptop'
        self._bbox = (0,0,1,1)
        self._conf = 0.0
        self._input = -1
        self._output = -1
        self._command=-1
        self._ip_add=""
        
    def set(self, name, bbox, conf):
        self._name = name
        self._bbox = bbox   # deep copy? 
        self._conf = conf
            
    # Given the center of the finger tips, find the relative location in the device bbox.
    # this is used for selection of modality.
    # If the hand is in the bbox, it returns a value of x & y in the range [0.1], otherwize returns (-1,-1)
    # ----------------------------------------------------------------------------------
    def relative_position(self, hx, hy)->list[float]:
        (x, y, x2, y2) = self._bbox
        if hx<x or hy<y or hx>x2 or hy>y2:
            return (-1,-1)
        return [(float(hx)-float(x))/(float(x2) - float(x)), (float(hy)-float(y))/(float(y2) - float(y))]
    
    # ----------------------------------------------------------------------------------
    def color(self):
        if self._name == 'tv':
            return (255, 255, 255) # white
        if self._name == 'laptop':
            return (255, 255, 0) # yellow
        if self._name == 'mouse':
            return (100, 100, 100) # gray
        if self._name == 'remote': 
            return (255, 000, 255) # purple
        if self._name == 'keyboard':
            return (100, 255, 100) # green
        if self._name == 'cell phone':
            return (100, 255, 255) # Turcoise                
        if self._name == 'microwave':
            return (255, 128, 128)   
        if self._name == 'book':
            return (255, 255, 180) # bright yellow 
        if self._name == 'clock':
            return (255, 0, 128)       
        return (0,0,0)
                     
    def print(self): 
        print("device: ", self._name, " bbox: ", self._bbox, " conf ", self._conf)   
    
    # ------------------------------------------------------------------------
    def draw(self, img):
        if (self._conf==0.0):
            return     
        (x, y, x2, y2) = self._bbox
        color = self.color()
        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
        cv2.putText(img, f"{self._name}: {self._conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    # ------------------------------------------------------------------------
    def draw_hand_location(self, img, hand_x, hand_y):
        if (self._conf==0.0):
            return     
        (x, y, x2, y2) = self._bbox
        color = self.color()
        [rx, ry] = self.relative_position(hand_x, hand_y)
        if rx>=0 and ry>=0:
            col = int(x + rx * (x2-x))
            row = int(y + ry * (y2-y))
            cv2.line(img, (x, row), (x2, row), color, 2)
            cv2.line(img, (col, y), (col, y2), color, 2)
                
                
                
# list of detected devices
# ===============================================================================================
class ListOfDevices:              
                
    def __init__(self):
        self._devices = []
    
    #
    # set the devices used for the demo.
    #
    def start(self):
        laptop = Device()
        laptop._name = 'laptop'
        laptop._ip_add='127.0.0.1'
        self._devices.append(laptop)
        
        phone = Device()
        phone._name = 'cell phone'
        phone._ip_add='127.0.0.1'
        self._devices.append(phone)
        
        mouse = Device()
        mouse._name = 'mouse'
        mouse._ip_add='127.0.0.1'
        self._devices.append(mouse)
    
    def reset_conf(self):
        for d in self._devices:
            d._conf = 0.0
              
    #
    # if the set for this experiment contains this class - return it
    # otherwise, return None
    #
    def update_exist(self, name:str, bbox, conf:float)->Device:
        for d in self._devices:
            if d._name==name:
                d._bbox = bbox
                d._conf = conf
                return d
        return None
        
    def append(self, d):
        if not (d is None):
            self._devices.append(d)
                
    def draw(self, img):
        for d in self._devices:
            if d._conf>0.0:
                d.draw(img)         
                
    def first_touched(self, x, y):
        for d in self._devices:
            if d._conf>0.0:
                [hx,hy] = d.relative_position(x, y)
                if hx>=0 and hy>=0: 
                    return [d,hx,hy]    
        return [None,-1,-1]   
    
    # Return another detection of class 'cls' that the current detection 'd' lies in it.
    # if none, returns None.
    # -------------------------------------------------------------------------------------------
    def find_overlapping_detection(self, d, cls:str='')->Device:
        (x, y, x2, y2) = d._bbox        
        for p in self._devices:
            if cls == '' or p.name == cls:
                (lx, ly, lx2, ly2) = p._bbox
                if lx<=x and lx2>= x2 and ly<=y and ly2>=y2:
                    return p
        return None
    
    # Reneter the first detection of class 'cls'. if none, returns None.
    # -------------------------------------------------------------------------------------------
    def find_next_detection(self, cls='')->Device:     
        for p in self._devices:
            if cls == '' or p.name == cls:
                return p
        return None
                            
# ===============================================================================================
class SingleCameraObserver:
    """
    Observer that shrinks the image, prints out its shape,
    and optionally sends the shrunk frame over UDP.
    """
    
     
    # ---------------------------------------------------------------
    def __init__(self, icons_list:list):
        self._udp_socket = None
        self._remote_ip = "127.0.0.1"
        self._remote_port = 8899
        self._shrink_factor = 0.25

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hand = mp.solutions.hands.Hands()
        
        self.hand_state = hand_pose.MID
        
        # Define configuration constants
        self._CONFIDENCE_THRESHOLD_LIMIT = 0.6
        self._BOX_COLOUR = (0, 255, 0)

        # Load the YOLO model
        print("Read Yolo Model ---------------")
        self._yolo_model = YOLO("./Models/yolo11n.pt")
        self._devices = ListOfDevices()
        self._devices.start()
        self._devices.reset_conf()
        
        if len(icons_list)>=3:
            self.icons={
                hand_pose.GRABBING:icons_list[0], 
                hand_pose.MID:icons_list[1],
                hand_pose.FULL_OPEN:icons_list[2],
            }
            print("dictionary size =", len(self.icons))
        else:
            print("SingleCameraObserver:__init__: ERROR: Failed to recice a list of 3 hand icons.")

    # ------------------------------------------------------------------------
    def set(self, udp_socket=None, remote_ip="127.0.0.1", remote_port=8899, shrink_factor=0.25):
        self._udp_socket = udp_socket
        self._remote_ip = remote_ip
        self._remote_port = remote_port
        self._shrink_factor = shrink_factor

    # ------------------------------------------------------------------------
    def draw_hand_state(self, ind, pink, thumb, p_rt, ind_rt, cnt, color, img:np.ndarray=None):
        
        h,w,_ = img.shape
        
        # finger tips triangle
        # cv2.line(img, ind, pink, (255,0,0), 2)
        # cv2.line(img, thumb, pink, (255,0,0), 2)
        # cv2.line(img, ind, thumb, (255,0,0), 2) 
                
        # # scaling distance
        # cv2.line(img, p_rt, ind_rt, (0,0,255), 2)
        
        # state
        #cv2.circle(img, cnt, 20, color, 2)    
        
        # icon
        if len(self.icons)>=3:
            sicon = cv2.resize(self.icons[self.hand_state], None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ih, iw,_ = sicon.shape
            st_x, st_y = cnt[0]-int(iw/2), cnt[1]-int(ih/2)
            end_x, end_y = st_x + iw, st_y + ih
            
            # if the icon image is within the image area.
            if st_x>=0 and st_y>=0 and end_x<w and end_y<h:
                for color in range(0, 3):
                    ret, mask = cv2.threshold(sicon[:,:,color], 252, 255, cv2.THRESH_BINARY_INV)
                    img[st_y : end_y, st_x : end_x, color] = (255-img[st_y : end_y, st_x : end_x, color]) * cv2.bitwise_not(mask) + sicon[:,:,color] * mask  

    #
    # if the hand open - measured by the area defined by the finger tips
    # ------------from ultralytics import YOLO------------------------------------------------------------
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
        self.hand_state = hand_pose.FULL_OPEN
        if area<0.5:
            if area>=0.1:
               color = (0,255,0)
               self.hand_state = hand_pose.MID
            else:
                color = (0,0,255)
                self.hand_state = hand_pose.GRABBING
                   
        center = polygon.centroid
        if h>0 and w>0:
            self.draw_hand_state(ind, pink, thumb, p_rt, ind_rt, (int(center.x), int(center.y)), color, img)
        
        return self.hand_state

    # -------------------(x, y, x2, y2) = bboxnd the label on the frame. 
    # The color of the bounding box depends on the confidence
    # I gnore most of the detected classes.
    # ------------------------------------------------------------------------
    def draw_detection_box(self, img, bbox, conf, cls, draw:bool=False)->Device:
        (x, y, x2, y2) = bbox
        
        if cls < 62 or cls >=75:
            # ignore classes of person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,
            # fire hydrant,stop sign,parking meter,bench,giraffe,backpack,umbrella,handbag,tie,suitcase,
            # frisbee,skis",snowboard,sports ball,kite,baseball bat,baseball glove,skateboard, surfboard,
            # tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich, orange,
            # brocolli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, 
            # toilet, vase, scissors, teddy bear, hair drier and toothbrush
            #
            return None        
        name = self._yolo_model.names[cls]
        if name == 'toaster' or name == 'sink' or name == 'refrigerator' or name == 'oven':
            return None
        
        # use tv, laptop, mouse, remote, keyboard, cell phone, microwave, book, clock,
        if draw:
            color = (250, 66, 100)
            if conf < self._CONFIDENCE_THRESHOLD_LIMIT:
                return None
            if conf > 0.6:
                color = (37, 245, 75)
            elif conf > 0.3:
                color = (66, 100, 245)
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)
            cv2.putText(img, f"{name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        d = Device()
        d.set(name, bbox, conf)
        return d
    
    # ------------------------------------------------------------------------
    def stream_frame(self, img):
        #Stream result frame.
        if self._udp_socket:
            success, encoded = cv2.imencode(".jpg", img)
            if success:
                frame_jpeg = encoded.tobytes()
                try:
                    self._udp_socket.sendto(frame_jpeg, (self._remote_ip, self._remote_port))
                except Exception as e:
                    print(f"Error sending UDP packet: {e}") 
   
    # ------------------------------------------------------------------------   
    def yolo_detection(self, img)->ListOfDevices:
        h, w, _ = img.shape

        #
        # set all devices that needs to be detected for the demo to zero confidance.
        #
        self._devices.reset_conf()
        
        yolo_image = img.copy() 
        if not(self._yolo_model is None):
            
            result = self._yolo_model(yolo_image)[0]
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            confidence = np.array(result.boxes.conf.cpu(), dtype="float")

            # list of relevant detections
            for cls, bbox, conf in zip(classes, bboxes, confidence):
                
                #
                # for our demos, this detection is modified:
                # A. It only looks for devices that are part of the demo and ignore other classes.
                # B. Each such device has updated bounding box and conf.
                #
                name = self._yolo_model.names[cls]
                d = self._devices.update_exist(name, bbox, conf)
                if d is None:
                    print("OUT: ",name," with conf ",conf," NOT in the demo. devices len ", len(self._devices._devices))
                else:
                    print("IN: ",name," with conf ",conf," IN the demo.", len(self._devices._devices))
                    #devices.append(self.draw_detection_box(img, bbox, conf, cls, draw=False))
        return self._devices

    # ------------------------------------------------------------------------
    def on_image_received(self, image: np.ndarray)->np.ndarray:
        shrunk_image = cv2.resize(image, None, fx=self._shrink_factor, fy=self._shrink_factor, interpolation=cv2.INTER_AREA)

        h, w, _ = shrunk_image.shape
        
        devices = self.yolo_detection(shrunk_image)
        devices.draw(shrunk_image)
          
        #
        # Mediapipe hand tracking
        #
        result = self._hand.process(shrunk_image)
        
        if result.multi_hand_landmarks:
            
           for hand_landmarks in result.multi_hand_landmarks:
                root = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_root, index_tip = hand_landmarks.landmark[5], hand_landmarks.landmark[8]
                #middle_tip = hand_landmarks.landmark[12]
                #ring_tip = hand_landmarks.landmark[16]
                pinky_root, pinky_tip = hand_landmarks.landmark[17], hand_landmarks.landmark[20]
        
                center_tips_x = ((index_tip.x + pinky_tip.x + thumb_tip.x) * w)/3
                center_tips_y = ((index_tip.y + pinky_tip.y + thumb_tip.y) * w)/3
       
                #
                # Draw hand, only if it lies inside at least one of the device?
                #
                [d, hx, hy] = devices.first_touched(center_tips_x, center_tips_y)
                if not(d is None):
                    d.draw_hand_location(shrunk_image, hx, hy)
                    #self.draw_finger_tips(index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip, h, w, img=shrunk_image)            
                    self.is_hand_open(index_tip, pinky_tip, thumb_tip, pinky_root, index_root, root, h, w, shrunk_image)
        
        
        return shrunk_image

#
# ===============================================================================================
class AriaVisualizer:

    def __init__(self, icons_list:list):
        self.plots = fpl.Figure(shape=(1, 2), names=[["frame", "detection"]], size=(1200, 800) )
        # fpl.GridPlot(shape=(1, 2), size=(1200, 800) )
            

        self.image_plot = {
            0: self.plots[0, 1].add_image(np.zeros((352, 352, 3), dtype="uint8"), vmin=0, vmax=255, ),
            aria.CameraId.Rgb: self.plots[0, 0].add_image(
                np.zeros((1408, 1408, 3), dtype="uint8"), vmin=0, vmax=255, ),
        }
        self.singleCameraObserver = SingleCameraObserver(icons_list=icons_list)
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
# ===============================================================================================
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
