import os
import typing
import logging

import cv2
import numpy as np
import stow

logger = logging.getLogger(__name__)

WINDOW_NAME = 'CameraCheckin'

class Engine:
    """Object to process webcam stream, video source or images
    All the processing can be customized and enchanced with custom_objects
    """
    def __init__(
        self,
        image_path: str = "",
        webcam_id: int = 0,
        show: bool = False,
        flip_view: bool = False,
        custom_objects: typing.Iterable = [],
        output_extension: str = 'out',
    ) -> None:
        """Initialize Engine object for further processing

        Args:
            image_path (str, optional): _description_. Defaults to "".
            video_path (str, optional): _description_. Defaults to "".
            webcam_id (int, optional): _description_. Defaults to 0.
            show (bool, optional): _description_. Defaults to False.
            flip_view (bool, optional): _description_. Defaults to False.
            custom_objects (typing.Iterable, optional): _description_. Defaults to [].
            output_extension (str, optional): _description_. Defaults to 'out'.
            start_video_frame (int, optional): _description_. Defaults to 0.
            end_video_frame (int, optional): _description_. Defaults to 0.
            break_on_end (bool, optional): _description_. Defaults to False.
        """
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects
        self.output_extension = output_extension
        self.width = 1280
        self.height = 720

    def flip(self, frame: np.ndarray) -> np.ndarray:
        """Flip given frame horizontally
        Args:
            frame: (np.ndarray) - frame to be flipped horizontally
        Returns:
            frame: (np.ndarray) - flipped frame if self.flip_view = True
        """
        if self.flip_view:
            return cv2.flip(frame, 1)

        return frame

    def open_window(self, width: int , height: int):
        """Open CV2 Window
        Put title window

        Args:
            width (int): _description_
            height (int): _description_
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, width, height)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson Nano')

    def display(self, frame: np.ndarray, webcam: bool = False, waitTime: int = 1) -> bool:
        """Display current frame if self.show = True
        When displaying webcam you can control the background images
        Args:
            frame: (np.ndarray) - frame to be displayed
            webcam: (bool) - Add additional function for webcam. Keyboard 'a' for next or 'd' for previous
        Returns:
            (bool) - Return True if no keyboard "Quit" interruption
        """
        if self.show:
            cv2.imshow(WINDOW_NAME, frame)
            k = cv2.waitKey(waitTime)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False

        return True

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with custom objects (custom object must have call function for each iteration)
        Args:
            frame: (np.ndarray) - frame to apply custom processing to
        Returns:
            frame: (np.ndarray) - custom processed frame
        """
        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def process_image(
        self,
        image: typing.Union[str, np.ndarray] = None,
        output_path: str = None
    ) -> np.ndarray:
        """The function does to processing with the given image or image path
        Args:
            frame: (typing.Union[str, np.ndarray]) - we can pass whether an image path or image numpy buffer
            output_path: (str) - we can specify where processed image will be saved
        Returns:
            frame: (np.ndarray) - final processed image
        """
        if image is not None and isinstance(image, str):
            if not stow.exists(image):
                raise Exception(f"Given image path doesn't exist {self.image_path}")
            else:
                extension = stow.extension(image)
                if output_path is None:
                    output_path = image.replace(f".{extension}", f"_{self.output_extension}.{extension}")
                image = cv2.imread(image)

        image = self.custom_processing(self.flip(image))

        cv2.imwrite(output_path, image)

        self.display(image, waitTime=0)
        return image

    def process_webcam(self, return_frame: bool = False) -> typing.Union[None, np.ndarray]:
        """Process webcam stream for given webcam_id
        """
        # Create a VideoCapture object for given webcam_id
        self.open_window(self.width, self.height)
        cap = cv2.VideoCapture(self.webcam_id)
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                print("Ignoring empty camera frame.")
                continue

            if return_frame:
                break

            frame = self.custom_processing(self.flip(frame))

            if not self.display(frame, webcam=True):
                break

        else:
            raise Exception(f"Webcam with ID ({self.webcam_id}) can't be opened")

        cap.release()
        return

    def run(self) -> None:
        if self.image_path:
            self.process_image(self.image_path)
        else:
            self.process_webcam()
