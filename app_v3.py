import streamlit as st
import cv2
import numpy as np
from queue import Queue
from threading import Thread
import time
import face_tool
from scrfd import SCRFD
from fps_metric import FPS

# Initialize face detector
face_detector = SCRFD(model_file='/Users/dominhtuan/Downloads/AnNinhSoiChieu/scrfd_10g_bnkps.onnx')

def get_vector_img(image, theshold_w=0.4, theshold_h=0.2):
    face_boxes, kpss = face_detector.detect(image, theshold_w=theshold_w, theshold_h=theshold_h, thresh=0.2)
    if face_boxes is not None:
        face_boxes = face_boxes.astype(np.int32)
        face_boxes = face_boxes[:, :4]
        face_boxes = np.maximum(face_boxes, 0)
    face_list = []
    if face_boxes is not None:
        for (x1, y1, x2, y2) in face_boxes:
            face_list.append((x1, y1, x2, y2))
        face_encoding = face_tool.face_encoding(image=image, kpss=kpss)

        return face_list, face_encoding
    return None, None

def recognize_face(image, kpss, known_faces):
    face_encodings = face_tool.face_encoding(image=image, kpss=kpss) if known_faces else []
    face_names = [int(face_tool.compare_faces(known_faces, enc, 0.6)[np.argmin(face_tool.face_distance(known_faces, enc))]) for enc in face_encodings]
    return face_names

def worker(input_q, output_q):
    while True:
        frame, known_faces, kpss = input_q.get()
        output_q.put({"status": recognize_face(frame, kpss, known_faces)})

# Load reference image
image_cc = cv2.imread("/Users/dominhtuan/Downloads/anhcccdtinh.jpg")
_, face_encoding_cc = get_vector_img(image_cc, theshold_w=0.05, theshold_h=0.05)

# Setup Streamlit
notification_container = st.empty()
cap1, cap2 = cv2.VideoCapture(1), cv2.VideoCapture(0)
col1, col2 = st.columns([4, 2])
placeholder1, placeholder2 = col1.empty(), col2.empty()
next_button = st.button("Next")

# Queue and threading
input_q, output_q = Queue(4), Queue()
Thread(target=worker, args=(input_q, output_q), daemon=True).start()
fps, frame_index, flag, list_frame_pass, frame_count, start_time = FPS().start(), 0, False, [], 0, time.time()

# Main loop
while cap1.isOpened() and cap2.isOpened():
    frame_index += 1
    if next_button:
        flag, list_frame_pass = False, []
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    if frame_index % 20 == 0:
        face_boxes, face_encoding_cc = get_vector_img(frame2, theshold_w=0.05, theshold_h=0.05)
        if not face_encoding_cc:
            cv2.putText(frame2, "KHONG TIM THAY ANH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        face_boxes, kpss = face_detector.detect(frame1)
        face_names = []
        if face_boxes is not None and kpss is not None and not input_q.full():
            face_boxes = face_boxes.astype(np.int32)
            face_boxes = face_boxes[:, :4]
            face_boxes = np.maximum(face_boxes, 0)
            input_q.put((frame1, face_encoding_cc, kpss))

        if output_q.qsize() > 0:
            face_names = output_q.get()['status']

        if face_boxes is not None:
            for (x1, y1, x2, y2), name in zip(face_boxes, face_names):
                color, text = ((0, 255, 0), "PASS") if name else ((0, 0, 255), "FAIL")
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame1, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if name:
                    list_frame_pass = [frame1, frame2]

    
    fps_value = frame_count / (time.time() - start_time)
    # start_time, frame_count = time.time(), 0
    cv2.putText(frame1, f'FPS: {fps_value:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_count += 1
    fps.update()
    placeholder1.image(frame1, channels="BGR", use_column_width=True)
    placeholder2.image(frame2, channels="RGB", use_column_width=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
