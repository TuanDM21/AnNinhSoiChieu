import cv2
import glob
import numpy as np
import face_tool
import os
from tqdm import tqdm
from PIL import Image
import typing
from PIL import ExifTags, Image
import face_tool


def PILtocv2(image_PIL):
    try:
        image_PIL = image_PIL.save('image.jpg')
    except:
        image_PIL = image_PIL.convert('RGB')
        image_PIL = image_PIL.save('image.jpg')
    return cv2.imread('image.jpg')

def encode_face(face_images: typing.List[str]) -> typing.List[np.ndarray]:
    embs = []
    embs_2 = []
    face_more_than_one_face = False
    for face_image in face_images:
        image = cv2.imread(face_image)

        # # rotate when smartphone
        # try:
        #     for orientation in ExifTags.TAGS.keys():
        #         if ExifTags.TAGS[orientation] == 'Orientation':
        #             break

        #         exif = dict(image._getexif().items())

        #     if exif[orientation] == 3:
        #         image = image.rotate(180, expand=True)
        #     elif exif[orientation] == 6:
        #         image = image.rotate(270, expand=True)
        #     elif exif[orientation] == 8:
        #         image = image.rotate(90, expand=True)
        # except (AttributeError, KeyError, IndexError):
        #     # cases: image don't have getexif
        #     pass

        # image = PILtocv2(image)
        print(f"Image Shape {image.shape}")
        encode = face_tool.face_encoding(image)
        try:
            if len(encode) > 1:
                face_more_than_one_face = True
                embs_2.append(encode[1])
            embs.append(encode[0])
        except:
            print('Dont have any face in this picture')
            pass
    if len(embs) > 0:
        print(f"Predict: {len(embs)}!")
        print(f"Embedding Face Length: {len(embs[0])}")

    if face_more_than_one_face:
        return [embs, embs_2]
    else:
        return [embs]

def main():
    folders = os.listdir('./image_data/')
    np_files = os.listdir('./biometric_data_3/')
    if not os.path.exists('./biometric_data_3/'):
        os.mkdir('./biometric_data_3/')

    failed = []

    print(np_files)
    # folders = ['6788']
    for folder in folders:
        # if folder + '.npy' in np_files:
        #     continue
        print('-- Folder {} --'.format(folder))
        files = glob.glob('./image_data/{}/*'.format(folder))
        DK_1 = True
        DK_2 = True
        try:
            feature_vectors = encode_face(face_images=files)
            DK_1 = len(feature_vectors) > 1 or len(feature_vectors) == 0
            if len(feature_vectors[0]) < 5 and len(feature_vectors[0]) >= 1:
                while len(feature_vectors[0]) < 5:
                    feature_vectors[0].append(feature_vectors[0][-1])
            DK_2 = len(feature_vectors[0]) != 5
            if DK_1 or DK_2:
                failed.append(folder)
                continue
            path = os.path.join('./biometric_data_3', folder)
            np.save(path, feature_vectors[0])
        except:
            continue
        print(failed)
    print(failed)

main()