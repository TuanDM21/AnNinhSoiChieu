import concurrent.futures
import logging
import os
import typing

import cv2
import numpy as np
import onnxruntime as ort

from utils import get_faces, hard_nms, preprocess_img, save_face_img

logger = logging.getLogger(__name__)


def predict_face(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)

    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


class FaceDetector:
    def __init__(
        self,
        onnx_model_path: str = "pretrain_model/ultra_light_640.onnx",
        face_confidence: float = 0.95,
        step: int = 1
    ) -> None:
        """Face Detector

        Args:
            onnx_model_path (str, optional): _description_. Defaults to "pretrain_model/ultra_light_640.onnx".
            anchors (typing.Union[str, dict], optional): _description_. Defaults to 'faces'.
        """
        if not os.path.exists(onnx_model_path):
            raise Exception(f"Model doesn't exists in {onnx_model_path}")

        self.ort_sess = ort.InferenceSession(onnx_model_path)
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.input_shape = self.ort_sess._inputs_meta[0].shape[2:4]

        self.face_confidence = face_confidence
        self.frame_index = 0
        self.step = step

    def detect(self, image: np.ndarray) -> typing.Tuple[typing.List[np.ndarray], typing.List[typing.Tuple]]:
        """Run object detection

        Args:
            image (np.ndarray): _description_

        Returns:
            typing.Tuple[typing.List[np.ndarray], typing.List[typing.Tuple]]: Result
        """
        img = preprocess_img(image)
        h, w, _ = image.shape
        confidences, boxes = self.ort_sess.run(None, {self.input_name: img})
        face_locations_, _, _ = predict_face(w, h, confidences, boxes, self.face_confidence)
        faces, face_locations = get_faces(face_locations_, image)

        # Filter of small face
        for idx, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            if (bottom - top)/h < 0.4 or (right - left)/w < 0.2:
                face_locations.remove(face_locations[idx])
                # faces.remove(faces[idx])

        return faces, face_locations

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Process multiple frames

        Args:
            frame (np.ndarray): _description_

        Returns:
            np.ndarray: Result
        """
        # Check frame idx to process
        pre_name = {}
        pre_name["state"] = False
        pre_name["name"] = "Unknown"
        face_locations = []
        face_names = []
        if self.frame_index == 0 or self.frame_index == self.step:
            face_names = []
            faces, face_locations = self.detect(frame)
            for i in range(len(face_locations)):
                face_names.append("Test")

            self.frame_index = 0

        # Display and send check-in/out request to server
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            r = {}
            if name == "Unknown":
                r['success'] = False
            else:
                if name == pre_name["name"] and pre_name["state"] is True:
                    r['success'] = True
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # future = executor.submit(check_in_out, cico_api, name)
                        future = None
                        try:
                            # r = future.result()
                            # r = check_in_out(cico, name)
                            r['success'] = True
                            pre_name["state"] = r['success']
                        except:
                            r['success'] = False

            font = cv2.FONT_HERSHEY_COMPLEX

            if r['success']:
                if self.frame_index == 0:
                    # save face to image
                    face = frame[top:bottom, left:right]
                    save_face_img(face, name)

                # print face
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

            pre_name["name"] = name

        self.frame_index += 1
        return frame
