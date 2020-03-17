# Code from: https://stackoverflow.com/questions/55169645/square-detection-in-image

import cv2
import numpy as np
import os
from time import sleep

def cropped_select(img, x, y, h, scale):
    scale1 = scale
    scale2 = 1 - scale
    return img[y+int(scale1*h):y+int(scale2*h), x+int(scale1*h):x+int(scale2*h)]

def get_images():
    image = cv2.imread(os.path.join('SquareDetection','computer-drawn-samples.jpg'))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,120,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 600
    max_area = 10000
    image_number = 0
    images = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = cropped_select(thresh, x, y, h, 0.08)
            resized = cv2.resize(ROI, (10, 10), interpolation=cv2.INTER_AREA)
            thresh_resized = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)[1]
            images.append(cv2.threshold(resized, 30, 1, cv2.THRESH_BINARY)[1].T.flatten())
            cv2.imwrite('ROI_{}.png'.format(image_number), thresh_resized)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            image_number += 1

    # cv2.imshow('sharpen', sharpen)
    # cv2.imshow('close', close)
    # cv2.imshow('thresh', thresh)
    cv2.imshow('image', image)
    cv2.waitKey(1000)
    cv2.waitKey(500)

    return images
