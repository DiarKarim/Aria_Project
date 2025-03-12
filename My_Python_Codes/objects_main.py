import cv2
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt



start = 30

# Histogram Back projection object
back_proj = HistHS()

# kernel filter
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

roi_hist = None
back_roi = None

while True:
    ret, frame = cam.read()

    if ret:
        small_fr = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        small_hsv = cv2.cvtColor(small_fr, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV', small_hsv)

        h, w, nc = small_fr.shape

        st_x = 0
        end_x = 50
        st_y = h -100
        end_y = h-1
        back_roi = small_hsv[st_y:end_y, st_x:end_x]

        cv2.rectangle(small_fr, (st_x, st_y), (end_x, end_y), (255, 255, 255), 2)
        cv2.imshow('Camera', small_fr)
        cv2.imshow('ROI', back_roi)

        roi_hist = cv2.calcHist([back_roi], [0, 1], None, [8, 8], [0, 181, 0, 256])
        min_val, max_val, _, _ = cv2.minMaxLoc(roi_hist)
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        n_min_val, n_max_val, _, _ = cv2.minMaxLoc(roi_hist)
        print("Hist: min ", min_val, "max", max_val, "Normalized: min ", n_min_val, "max", n_max_val)
        dst = cv2.calcBackProject([small_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        cv2.imshow('Dst', dst)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.filter2D(dst, -1, disc)



        # ROI


        # small_bw = cv2.cvtColor(small_fr, cv2.COLOR_BGR2GRAY)
        # blur_bw = cv2.GaussianBlur(small_bw, (5, 5), 1)
        # canny = cv2.Canny(blur_bw, 30, 200)
        # contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        # black = np.zeros(small_fr.shape, np.uin
        #         # bigCntrs = []
        #         # cntrRect = []
        #         # for c in contours:
        #         #
        #         #     # big enough?
        #         #     area = cv2.contourArea(c)
        #         #     if area > 100:
        #         #         bigCntrs.append(c)
        #         #
        #         #         epsilon = 0.05 * cv2.arcLength(c, True)
        #         #         approx = cv2.approxPolyDP(c, epsilon, True)
        #         #         if len(approx) < 6:
        #         #             cntrRect.append(approx)
        #         #
        #         # cv2.drawContours(black, bigCntrs, -1, (255, 255, 255), -1)
        #         # cv2.drawContours(black, cntrRect, -1, (0, 255, 0), -1)
        #         #
        #         # #out.write(frame)
        #         # disp = cv2.hconcat(small_fr, black)
        #         #
        #         # cv2.imshow('Contours', black)t8)
        #

    K = cv2.waitKey(1)
    if K == ord('q') or K==27:
        break

cam.release()
#out.release()
cv2.destroyAllWindows()