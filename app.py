import streamlit as st
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from typing import Dict, List
from PIL import Image
import time
import cv2
import face_tool
from face_model import FaceModel
from fps_metric import FPS
from scrfd import SCRFD
from utils import check_in_out, get_data, maybe_create_folder, open_cam_usb
from web_cam import WebcamVideoStream
import glob
import face_tool
face_detector = SCRFD(model_file='/Users/dominhtuan/Downloads/AnNinhSoiChieu/scrfd_10g_bnkps.onnx')
face_model = FaceModel(onnx_model_path='/Users/dominhtuan/Downloads/AnNinhSoiChieu/webface_r50.onnx')



def dialog_pass():
    dial = st.dialog("Warning")
    dial.write("PASS")

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

image_cc = cv2.imread("/Users/dominhtuan/Downloads/anhcccdtinh.jpg")
face_list,face_encoding_cc = get_vector_img(image_cc,theshold_w=0.05,theshold_h=0.05)

def recognize_face(
        image: np.ndarray,
        kpss: np.ndarray,
        known_faces: List[np.ndarray],
) -> List[str]:
    if not known_faces:
        return [0]
    face_encodings = face_tool.face_encoding(image=image, kpss=kpss)
    face_names = []
    # Face recognition
    for face_encoding in face_encodings:
        matches = face_tool.compare_faces(known_faces, face_encoding, tolerance=0.6)
        face_distances = face_tool.face_distance(known_faces, face_encoding)
        best_idx = np.argmin(face_distances)
        if matches[best_idx]:
            face_names.append(1)
        else:
            face_names.append(0)
    return face_names

def worker(input_q: Queue, output_q: Queue):
    while True:

        frame, known_faces,kpss = input_q.get()
        
        status = recognize_face(
                image=frame,
                kpss=kpss,
                known_faces=known_faces
            )
        output = {"status": status}
        output_q.put(output)

notification_container = st.empty()

# OpenCV VideoCapture objects for two webcams
cap1 = cv2.VideoCapture(1)  # Change the index if needed
cap2 = cv2.VideoCapture(0)  # Change the index if needed

# Create two columns for displaying placeholders side by side
col1, col2 = st.columns([4,2])
# Create two empty placeholders
placeholder1 = col1.empty()
placeholder2 = col2.empty()

next_button = st.button("Next")

frame_index = 0
face_step = 2
detect_step = 2
reset_step = 120

text = "KHONG TIM THAY ANH"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_thickness = 2
text_color = (0, 0, 255)

input_q = Queue(4)  # fps is better if queue is higher but then more lags
output_q = Queue()
face_encoding_cc = None
for i in range(1):
    t = Thread(target=worker, args=(input_q, output_q))
    t.daemon = True
    t.start()

fps = FPS().start()
flag = False
list_frame_pass = []
fps_new = 0
frame_count = 0
start_time = time.time()

while cap1.isOpened() and cap2.isOpened():
    frame_index +=1
    if next_button:
        flag = False
        list_frame_pass = []
    if flag == False:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # if not ret1 or not ret2:
        if not ret1:
            st.error("Error: Failed to capture frame 1.")
            break
        if not ret2:
            st.error("Error: Failed to capture frame 2.")
            break

        # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = cv2.flip(frame1, 1)
        if frame_index % detect_step == 0:
            face_list, face_encoding_cc = get_vector_img(frame2, theshold_w=0.05, theshold_h=0.05)
            if not face_encoding_cc:
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_position = (frame2.shape[1] - text_size[0], text_size[1] + 10)
                cv2.putText(frame2, text, text_position, font, font_scale, text_color, font_thickness)

        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        face_list = []
        face_names = []
        if frame_index % detect_step == 0:
            face_boxes, kpss = face_detector.detect(frame1)
            if face_boxes is not None:
                face_boxes = face_boxes.astype(np.int32)
                face_boxes = face_boxes[:, :4]
                face_boxes = np.maximum(face_boxes, 0)
                for (x1, y1, x2, y2) in face_boxes:
                    face_names.append(0)
                    face_list.append((x1, y1, x2, y2))
                    img_save = frame1[y1:y2, x1:x2]
                    number_of_images = len(glob.glob('/Users/dominhtuan/Downloads/AnNinhSoiChieu/image_save/*'))
                    if number_of_images < 4:
                        cv2.imwrite(f'image_save/output_image_{number_of_images + 1}.jpg', img_save)
            if kpss is not None :
                if not input_q.full():
                    input_q.put((frame1,face_encoding_cc, kpss))

        if output_q.empty():
            pass  # fill up queue
        else:
            data = output_q.get()
            face_names = data['status']

        # if frame_index % (reset_step * 10) == 0:
        #     checked_name = {}
        #     frame_index = 0
        font = cv2.FONT_HERSHEY_COMPLEX
        print(face_list,face_names)
        for (x1, y1, x2, y2), name in list(zip(face_list, face_names)):
            top = y1
            bottom = y2
            left = x1
            right = x2
            notification_container.empty()
            if name:
                next_button = False
                flag = True
                # st.write("This is a simple text message.")
                # st.success("Operation was successful!")
                notification_container.success("Operation was successful!")
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame1, "{}".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                list_frame_pass = [frame1,frame2]
            else:
                # st.error("Error: An error occurred.")

                notification_container.error("Error: An error occurred.")
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                if name == "Unknown":
                    cv2.putText(frame1, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                else:
                    cv2.putText(frame1, "{}".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        frame_count += 1

        # Calculate FPS every second
        if frame_count == 10:
            end_time = time.time()
            time_diff = end_time - start_time
            fps_new = frame_count / time_diff
            start_time = end_time
            frame_count = 0
        cv2.putText(frame1, f'FPS: {fps_new:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        frame1 = list_frame_pass[0]
        frame2 = list_frame_pass[1]

    fps.update()
    # Check for key press to exit the loop
    # Update the placeholders with the frames
    placeholder1.image(frame1, channels="BGR", use_column_width=True)
    placeholder2.image(frame2, channels="RGB", use_column_width=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close Streamlit app
cap1.release()
cap2.release()
cv2.destroyAllWindows()
