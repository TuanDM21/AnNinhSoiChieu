import argparse
import concurrent.futures
import logging
import os
from queue import Queue
from threading import Thread
from typing import Dict, List
from PIL import Image
import cv2
import numpy as np

import face_tool
from face_model import FaceModel
from fps_metric import FPS
from scrfd import SCRFD
from utils import check_in_out, get_data, maybe_create_folder, open_cam_usb
from web_cam import WebcamVideoStream
import glob

face_detector = SCRFD(model_file='D:\Check-In-Check-Out-Jetson-Nano\Check-In-Check-Out-Jetson-Nano\scrfd_10g_bnkps.onnx')
face_model = FaceModel(onnx_model_path='D:\Check-In-Check-Out-Jetson-Nano\Check-In-Check-Out-Jetson-Nano\webface_r50.onnx')

width = 1280
height = 720
WINDOW_NAME = 'CameraCheckIn'

def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson Nano')


def worker(input_q: Queue, output_q: Queue):
    while True:
        frame, kpss = input_q.get()
        names = "MR.Ki"
        output = {"names": names}
        output_q.put(output)

def start_app(video_capture):
    frame_index = 0
    face_step = 30
    detect_step = 15
    reset_step = 120

    input_q = Queue(2)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    fps = FPS().start()
    while True:
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        face_list = []
        if frame_index % detect_step == 0:
            face_boxes, kpss = face_detector.detect(frame)
            if face_boxes is not None:
                face_boxes = face_boxes.astype(np.int32)
                face_boxes = face_boxes[:, :4]
                face_boxes = np.maximum(face_boxes, 0)
                for (x1, y1, x2, y2) in face_boxes:
                    face_list.append((x1, y1, x2, y2))
                    img_save = frame[y1:y2,x1:x2]
                    # imageq = Image.fromarray(img_save)
                    number_of_images = len(glob.glob('D:\Check-In-Check-Out-Jetson-Nano\Check-In-Check-Out-Jetson-Nano\image_save\*'))
                    if  number_of_images < 4:
                        cv2.imwrite(f'image_save\output_image_{number_of_images +1}.jpg', img_save)
                    # pass

            if kpss is not None and frame_index % face_step == 0:
                if not input_q.full():
                    input_q.put((frame, kpss))

        if output_q.empty():
            pass  # fill up queue
        else:
            data = output_q.get()
            face_names = data['names']

        if frame_index % (reset_step * 10) == 0:
            checked_name = {}
            frame_index = 0

        font = cv2.FONT_HERSHEY_COMPLEX

        for (x1, y1, x2, y2) in face_list:
            top = y1
            bottom = y2
            left = x1
            right = x2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "{}".format("KI"), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow(WINDOW_NAME, frame)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        fps.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()

def main():
    print('OpenCV version: {}'.format(cv2.__version__))

    video_capture = open_cam_usb(0)
    open_window(width, height)

    start_app(video_capture)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
