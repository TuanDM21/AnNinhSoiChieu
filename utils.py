import errno
import os
import subprocess

import cv2
import numpy as np
import requests
from PIL import Image
import datetime



def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-suppression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def preprocess_img(frame, input_shape=(640, 480)):
    """
    Preprocessing Frame for predict faces boxes
    Input: frame
    Output: frame after preprocessing
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
    img = cv2.resize(img, input_shape) # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def get_faces(boxes, frame):
    """
    Crop frame to faces using boxes
    Args:
            boxes: bounding box
            frame: frame
    Return: list of faces in frame
            clean_boxes: list of ndarray
    """
    faces = []
    clean_boxes = []
    for i in range(boxes.shape[0]):
        top, left, bottom, right = boxes[i,:]
        face = frame[top:bottom, left:right]
        fix_box = (left, bottom, right, top)

        try:
            face = cv2.resize(face, (112, 112))
            faces.append(face)
            clean_boxes.append(fix_box)
        except:
            continue

    return faces, clean_boxes


def get_data(biometric_data: str):
    # os.system("rm -rf biometric_data")
    # os.system("mkdir biometric_data")
    print("Pulling data from aws")
    # Pull biometric data from aws
    # os.system('rm -rf biometric_data')
    os.system('mkdir biometric_data_sub')
    data = requests.get(url = biometric_data,verify = False)
    data = data.json()
    print("finish pulling")
    for i in range(len(data)):
        url_data = data[i]["biometricUrl"]
        print("url_data",url_data, data[i]["badgeNumber"])
        name = data[i]["badgeNumber"]
        if url_data:
            os.system('curl {} > "biometric_data_sub/{}.npy"'.format(url_data, name))

    os.system('rm -rf biometric_data')
    os.system('mv biometric_data_sub biometric_data')


def check_in_out(cico: str, name: str):
    json = {
        'key':'A189FB9E2682CCA1DDF93687E96A7',
        'badgeNumber':name,
        'sensorId':str(1)
    }
    r = requests.post(cico, json=json, verify=False)

    return r.json()


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(src: int = 0):
    return cv2.VideoCapture(src)


def open_cam_onboard(sensor_id,width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc sensor_id={} ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1280, height=(int)720, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(sensor_id,width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1280, height=(int)720, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def get_data_img(biometric_data):
    # os.system("rm -rf biometric_data")
    # os.system("mkdir biometric_data")
    print("Pulling data from aws")
    # Pull biometric data from aws
    os.system('rm -rf image_data')
    os.system('mkdir image_data')
    data = requests.get(url = biometric_data, verify = False)
    data = data.json()
    print("finish pulling")
    for i in range(len(data)):
        if i == 0 or i % 5 == 0:
            os.system('mkdir image_data/{}'.format(data[i]["badgeNumber"]))
        url_data = data[i]["biometricUrl"]
        print("url_data",url_data, data[i]["badgeNumber"])
        name = data[i]["badgeNumber"]
        if url_data:
            os.system('curl {} > "image_data/{}/{}.jpg"'.format(url_data, name, i%5))


def get_person_data(biometric_data, name_ID):
    os.system('rm -rf {}'.format(name_ID))
    os.system('mkdir {}'.format(name_ID))
    data = requests.get(url = biometric_data, verify = False)
    data = data.json()
    print("finish pulling")
    for i in range(len(data)):
        # if i == 0 or i%5 == 0:
        #     os.system('mkdir image_data/{}'.format(data[i]["badgeNumber"]))
        url_data = data[i]["biometricUrl"]
        # print("url_data", url_data, data[i]["badgeNumber"])
        name = data[i]["badgeNumber"]
        if url_data and name == name_ID:
            os.system('curl {} > "{}/{}.jpg"'.format(url_data, name, i%5))


def maybe_create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def save_face_img(face, name):
    """
    Save image face into folder
    """
    folder = './data_checkin_checkout/{}'.format(name)
    maybe_create_folder(folder=folder)
    _len_folder = len(os.listdir(folder))
    datetime_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if _len_folder < 100:
        image_path = os.path.join(folder, str(name)+ "_" + str(datetime_name) + '.jpg')
        cv2.imwrite(image_path, face)
