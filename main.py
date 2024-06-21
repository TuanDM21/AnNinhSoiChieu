import logging

from engine import Engine
from face_detection import FaceDetector
from fps_metric import FPSmetric

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


def main():
    engine = Engine(show=True, flip_view=True, webcam_id=0, custom_objects=[FaceDet eector(), FPSmetric()])
    engine.run()

main()

