import argparse
import concurrent.futures
import logging
import os
from queue import Queue
from threading import Thread
from typing import Dict, List

import cv2
import numpy as np

import face_tool
from face_model import FaceModel
from fps_metric import FPS
from scrfd import SCRFD
from utils import check_in_out, get_data, maybe_create_folder, open_cam_usb
from web_cam import WebcamVideoStream

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s:%(lineno)d] - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('test.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

WINDOW_NAME = 'CameraCheckIn'

biometric_data = 'https://tms.blueeye.ai/api/engine/biometrics?key=A189FB9E2682CCA1DDF93687E96A7'
cico_api = 'https://tms.blueeye.ai/api/engine/check-in-out'

face_detector = SCRFD(model_file='./pretrain_model/scrfd_10g_bnkps.onnx')
face_model = FaceModel(onnx_model_path='./pretrain_model/webface_r50.onnx')


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson Nano'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--width', dest='width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='height',
                        help='image height [720]',
                        default=720, type=int)
    parser.add_argument("-g", "--getdata", type=bool,
                        default=False,
                        help="choose to update biometric_data")
    args = parser.parse_args()
    return args


def check_encode_distance(know_face, unknown_face, threshold):
    self_face_distances = face_tool.face_distance(know_face, unknown_face)
    check_list = [distances < threshold for distances in self_face_distances]
    logger.info(check_list)
    logger.info(self_face_distances)
    if check_list.count(True) >= 3:
        return True
    else:
        return False


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson Nano')


def load_data(known_faces, name_faces, face_dict):
    known_faces = []
    name_faces = []
    face_dict = {}

    for data in os.listdir('biometric_data'):
        path = os.path.join('biometric_data', data)
        temp = np.load(path)
        name = os.path.splitext(data)[0]

        if len(temp[0]) != 512:
            logger.info(f"{name} not use new vector")
            continue

        name = os.path.splitext(data)[0]
        face_dict[name] = temp
        for known_faces_idx in temp:
            known_faces.append(known_faces_idx)
            name_faces.append(name)

    return known_faces, name_faces, face_dict


def update_data(known_faces, name_faces, face_dict):
    get_data("biometric_data")
    known_faces, name_faces, face_dict = load_data(known_faces, name_faces, face_dict)


def recognize_face(
    image: np.ndarray,
    kpss: np.ndarray,
    known_faces: List[np.ndarray],
    name_faces: List[str],
    face_dict: Dict
) -> List[str]:
    face_encodings = face_tool.face_encoding(image=image, kpss=kpss)
    face_names = []

    # Face recognition
    for face_encoding in face_encodings:
        matches = face_tool.compare_faces(known_faces, face_encoding, tolerance=0.6)
        face_distances = face_tool.face_distance(known_faces, face_encoding)

        best_idx = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_idx]:
            name = name_faces[best_idx]
            logger.info(f"{name} - {face_distances[best_idx]}")

        if name == "Unknown":
            face_names.append("Unknown")
        else:
            if check_encode_distance(face_dict[name], face_encoding, 0.6):
                face_names.append(name)
            else:
                face_names.append("Unknown")

    return face_names


def worker(input_q: Queue, output_q: Queue, known_faces, name_faces, face_dict):
    while True:
        frame, kpss = input_q.get()
        names = recognize_face(
            image=frame,
            kpss=kpss,
            known_faces=known_faces,
            name_faces=name_faces,
            face_dict=face_dict,
        )
        output = {"names": names}
        output_q.put(output)


def start_app(video_capture):
    # Load biometric data
    known_faces = []
    name_faces = []
    face_dict = {}
    known_faces, name_faces, face_dict = load_data(known_faces, name_faces, face_dict)

    input_q = Queue(2)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q, known_faces, name_faces, face_dict))
        t.daemon = True
        t.start()

    fps = FPS().start()
    pre_name = {}
    pre_name["state"] = False
    pre_name["name"] = "Unknown"
    name = "Unknown"

    frame_index = 0
    face_step = 30
    detect_step = 15
    reset_step = 120
    r = {}
    checked_name = {}

    while True:
        frame_index += 1
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
                    # face_list.append((x1, y1, x2, y2))
                    pass

            if kpss is not None and frame_index % face_step == 0:
                if not input_q.full():
                    input_q.put((frame, kpss))

        if output_q.empty():
            pass  # fill up queue
        else:
            data = output_q.get()
            face_names = data['names']
            for name in face_names:
                if name == "Unknown":
                    r['success'] = False
                else:
                    if name == pre_name['name'] and pre_name['state'] is True:
                        r['success'] = True
                    else:
                        if checked_name.get(name, False):
                            r['success'] = True
                            pre_name['state'] = r['success']
                            continue
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(check_in_out, cico_api, name)
                            try:
                                r = future.result()
                                pre_name['state'] = r['success']
                                logger.info(f"Tai {frame_index}/{reset_step}: Diem Danh Thanh Cong Cho {name}")
                            except:
                                r['success'] = False

                pre_name['name'] = name

        if frame_index % reset_step == 0:
            pre_name['name'] = "Unknown"
            pre_name["state"] = False
            r['success'] = False
            name = "Unknown"

        if frame_index % (reset_step * 10) == 0:
            checked_name = {}
            frame_index = 0

        font = cv2.FONT_HERSHEY_COMPLEX

        for (x1, y1, x2, y2) in face_list:
            top = y1
            bottom = y2
            left = x1
            right = x2
            if r.get('success', False) and name != "Unknown":
                checked_name[name] = True
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "{}".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                if name == "Unknown":
                    cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "{}".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if r.get('success', False) and name != "Unknown":
            checked_name[name] = True
            cv2.putText(frame, "Da Diem Danh - {}".format(name), (50, 50), font, 1.5, (0, 255, 0), 2)
        else:
            if name == "Unknown":
                cv2.putText(frame, "Unknown", (50, 50), font, 1.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Da Diem Danh - {}".format(name), (50, 50), font, 1.5, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        fps.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    logger.info('elapsed time (total): {:.2f}'.format(fps.elapsed()))
    logger.info('approx. FPS: {:.2f}'.format(fps.fps()))


def main():
    args = parse_args()
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))
    maybe_create_folder("./data_checkin_checkout")

    if args.getdata:
        get_data(biometric_data)

    # video_capture = WebcamVideoStream(args.video_source, args.width, args.height).start()
    video_capture = open_cam_usb(args.video_source)
    open_window(args.width, args.height)

    start_app(video_capture)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
