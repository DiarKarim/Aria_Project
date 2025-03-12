# Packages Import
import cv2
import os
import numpy as np

class BackROI:
    def __init__(self, st_x:int, st_y:int, end_x:int, end_y:int):
        self._st_x, self._st_y = st_x, st_y
        self._end_x, self._end_y = end_x, end_y

    def set(self, st_x:int, st_y:int, end_x:int, end_y:int):
        self._st_x = st_x
        self._st_y = st_y
        self._end_x = end_x
        self._end_y = end_y

    def image_roi(self, image)->list[bool, np.ndarray]:
        if (self._end_x>self._st_x and self._end_y>self._st_y):
            return [True, image[self._st_y:self._end_y, self._st_x:self._end_x]]
        else:
            return [False, None]
    def draw_roi(self, image, color=(255, 255, 255), width = 2)->cv2.Mat:
        if (self._end_x > self._st_x and self._end_y > self._st_y):
            cv2.rectangle(image, (self._st_x, self._st_y), (self._end_x, self._end_y), color, width)
        return image

    def print(self):
        print("X ", self._st_x, ":", self._end_x, " Y ", self._st_y, ":", self._end_y)

#
# Histogram background projection
# ------------------------------------------
class HistHS:

    def __init__(self):
        self._back_hist = None
        self._bins = 25
        self._ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    def calc_hsv(self, back_hsv)->cv2.Mat:
        roi_hist = cv2.calcHist([back_hsv], [0, 1], None, [8, 8], [0, 181, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX, -1)
        return roi_hist

    def back_project(self, img_hsv, roi_hist, thresh:int=50)->list[cv2.Mat,cv2.Mat]:
        bp = cv2.calcBackProject([small_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        bp_binary = cv2.threshold(bp, thresh, 255, cv2.THRESH_BINARY)[1]
        bp_median = cv2.medianBlur(bp_binary, 5)
        bp_mask = cv2.filter2D(bp_median, -1, self._ellipse_kernel)
        return [bp, bp_mask]

    def back_project_mask(self, img, disk_kernel)->np.ndarray:
        bp = self.back_project(img)
        dst = cv2.filter2D(bp, -1, disk_kernel)
        #_,mask = cv2.threshold(bp, 70, 255, 0)
        return dst

    def print_histogram_stat(self,roi_hist):
        n_min_val, n_max_val, _, _ = cv2.minMaxLoc(roi_hist)
        print("Normalized: min ", n_min_val, "max", n_max_val)

    def debug_draw(self):
        pass
        # if len(self._back_hist) == 1:
        #     plt.plot(self._back_hist, color='black')
        # else:
        #     if len(self._back_hist) == 3:
        #         plt.plot(self._back_hist[0], color='b')
        #         plt.plot(self._back_hist[1], color='g')
        #         plt.plot(self._back_hist[2], color='b')
        # plt.xlim([0, 256])
        # plt.show()




cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# kernel filter
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

saliency_obj = Saliency()

back_roi = BackROI(0, 150, 50, 250)
back_proj = HistHS()


roi_hist = None
back_roi_img = None

while True:
    ret, frame = cam.read()
    if ret:
        small_fr = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rows, cols, _ = small_fr.shape
        small_hsv = cv2.cvtColor(small_fr, cv2.COLOR_BGR2HSV)

        #
        # Update background sample
        #
        back_roi.set(0, rows -100, 50, rows-1)
        back_roi_img = back_roi.image_roi(small_hsv)[1]

        #cv2.imshow("Image", cv2.hconcat(([small_fr, back_roi.draw_roi(small_hsv, color=(255, 255, 255), width=2)])))
        #back_roi.display_sub_image(small_hsv, "ROI_subImage")

        roi_hist = back_proj.calc_hsv(back_roi_img)
        #back_proj.print_histogram_stat(roi_hist)

        dst, mask = back_proj.back_project(small_hsv, roi_hist, thresh=50)
        #cv2.imshow('Dst', cv2.hconcat([dst, mask]))




        saliency_obj.calc(small_fr)
        saliency_obj.display_combined(thresh=10)

        mask2 = cv2.bitwise_and(cv2.bitwise_not(mask), saliency_obj.fine_map)
        cv2.imshow('CombMask', mask2)

    K = cv2.waitKey(1)
    if K == ord('q') or K == 27:
        break

cam.release()
#out.release()
cv2.destroyAllWindows()