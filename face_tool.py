import traceback
from typing import List

import cv2
import numpy as np
from face_align import norm_crop
import logging

from face_model import FaceModel
from scrfd import SCRFD

detector = SCRFD(model_file='D:\Check-In-Check-Out-Jetson-Nano\Check-In-Check-Out-Jetson-Nano\scrfd_10g_bnkps.onnx')
model = FaceModel(onnx_model_path='D:\Check-In-Check-Out-Jetson-Nano\Check-In-Check-Out-Jetson-Nano\webface_r50.onnx')

logger = logging.getLogger(__name__)

def face_encoding(
    image: np.ndarray,
    kpss: List = None,
    file_name: str = None,
) -> List[np.ndarray]:
    if kpss is None:
        _, kpss = detector.detect(image)

    if kpss is None:
        kpss = []
        logger.info("No detected face")
    logger.info("Number of Faces: {}".format(len(kpss)))

    # Get result
    result = []
    for kps in kpss:
        try:
            norm_img = norm_crop(image, kps)
            if file_name is not None:
                cv2.imwrite(file_name, norm_img)
            feature_vector = model.encode(norm_img)
            result.append(feature_vector)
        except:
            logger.info(f"{traceback.print_exc()}")

    return result


def compare_faces(
    known_face_encodings: List[np.ndarray],
    face_encoding_to_check: np.ndarray,
    tolerance: float = 0.6
) -> List[bool]:
    return model.compare_faces(
        known_face_encodings=known_face_encodings,
        face_encoding_to_check=face_encoding_to_check,
        tolerance=tolerance
    )


def face_distance(
    face_encodings: List[np.ndarray],
    face_to_compare: np.ndarray,
) -> np.ndarray:
    return model.face_distance(face_encodings=face_encodings, face_to_compare=face_to_compare)
