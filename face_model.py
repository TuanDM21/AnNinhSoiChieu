import logging
import os
import typing

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class FaceModel:
    def __init__(
        self,
        onnx_model_path: str = "pretrain_model/webface_r50.onnx",
        anchors: typing.Union[str, dict] = 'faces',
        force_cpu: bool = True,
    ) -> None:
        """

        Args:
            onnx_model_path (str, optional): _description_. Defaults to "models/faceNet.onnx".
            anchors (typing.Union[str, dict], optional): _description_. Defaults to 'faces'.
            force_cpu (bool, optional): _description_. Defaults to False.

        Raises:
            Exception: _description_
        """
        if not os.path.exists(onnx_model_path):
            raise Exception(f"Model doesn't exists in {onnx_model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        providers = providers if ort.get_device() == "GPU" and not force_cpu else providers[::-1]

        self.ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

        self.input_shape = tuple(self.ort_sess._inputs_meta[0].shape[2:4])

        self.anchors = self.load_anchors(anchors) if isinstance(anchors, str) else anchors

    def load_anchors(self, faces_path: str):
        """Generate anchors for given faces path

        Args:
            faces_path: (str) - path to directory with faces

        Returns:
            anchors: (dict) - dictionary with anchor names as keys and anchor encodings as values
        """
        anchors = {}
        return anchors

    def cosine_distance(self, a: np.ndarray, b: typing.Union[np.ndarray, list]) -> np.ndarray:
        """Cosine distance between vectors a and b

        Args:
            a: (np.ndarray) - first vector
            b: (np.ndarray) - second list of vectors

        Returns:
            distance: (float) - cosine distance
        """
        if isinstance(a, list):
            a = np.array(a)

        if isinstance(b, list):
            b = np.array(b)

        return 1 - np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=-1))

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image

        Args:
            img: (np.ndarray) - image to be normalized

        Returns:
            img: (np.ndarray) - normalized image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mean, std = img.mean(), img.std()
        # return (img - mean) / std
        img = img / 255.
        return img

    def l2_normalize(self, x: np.ndarray, axis: int = -1, epsilon: float = 1e-10) -> np.ndarray:
        """l2 normalization function

        Args:
            x: (np.ndarray) - input array
            axis: (int) - axis to normalize
            epsilon: (float) - epsilon to avoid division by zero

        Returns:
            x: (np.ndarray) - normalized array
        """
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def encode(self, face_image: np.ndarray) -> np.ndarray:
        """Generate embedded vector by face model
        Args:
            face_image: input array
        Returns:
            np.array: feature vector
        """
        face = self.normalize(face_image)
        face_shape = face_image.shape[:2]
        if face_shape != self.input_shape:
            face = cv2.resize(face, self.input_shape).astype(np.float32)
        else:
            face = face.astype(np.float32)
        input_model = np.expand_dims(face, axis=0)  # Model Shape (1, width, heigh, 3)
        input_model = np.transpose(input_model, (0, 3, 1, 2))  # Permute input shape to (1, 3, width, height)

        encode = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: input_model})[0][0]
        normalized_encode = self.l2_normalize(encode)

        return normalized_encode

    def face_distance(
        self,
        face_encodings: typing.List[np.ndarray],
        face_to_compare: np.ndarray
    ) -> np.ndarray:
        """Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.

        Args:
            face_encodings (np.ndarray): List of face encodings to compare
            face_to_compare (np.ndarray): A face encoding to compare against

        Returns:
            np.ndarray: The list of distance
        """
        if len(face_encodings) == 0:
            return np.empty((0))
        distances = self.cosine_distance(face_to_compare, face_encodings)
        distances = np.round(distances, 3)
        return distances

    def compare_faces(
        self,
        face_encoding_to_check: np.ndarray,
        known_face_encodings: typing.List[np.ndarray],
        tolerance: float = 0.45
    ) -> typing.List:
        """
            Compare a list of face encodings against a candidate encoding to see if they match.
        Args:
            known_face_encodings (_type_): A list of known face encodings
            face_encoding_to_check (_type_):  A single face encoding to compare against the list
            tolerance (float, optional): How much distance between faces to consider it a match.
                                        Lower is more strict. Defaults to 0.15.
        Returns:
            typing.List:  A list of True/False values indicating
                        which known_face_encodings match the face encoding to check
        """
        return list(self.face_distance(
            face_encodings=known_face_encodings, face_to_compare=face_encoding_to_check) <= tolerance)
