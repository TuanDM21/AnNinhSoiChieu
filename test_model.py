import glob

import cv2
import numpy as np

import face_tool
from face_model import FaceModel
from face_tool import face_encoding

face = FaceModel()
img1 = cv2.imread('test.jpg')

# vector = face_encoding(image=img1, file_name='test.jpg')[0]
# print(vector.shape)
# vector = face.encode(img1)
# # print(vector)
# data = [vector] * 5


def gen_vector(folder):
    result = []
    files = glob.glob(folder, recursive=True)
    for file in files:
        img2 = cv2.imread(file)
        _vector = face_encoding(image=img2, file_name=file.split('/')[-1] + '.jpg')
        if len(_vector) > 0:
            result.append(_vector[0])
    return result


data = np.load('./biometric_data_3/6973.npy')
# print(data)
count = 0

# data = gen_vector('./6788/*')

files = glob.glob('/home/namhn89/data_checkin_checkout_20221221/6973/*.jpg')
for file in files:
    img = cv2.imread(file)
    _vector = face.encode(img)
    distances = face_tool.face_distance(data, _vector)
    matches = face_tool.compare_faces(data, _vector)
    print(distances)
    
    print(matches)
    if sum(matches) < 3:
        print(file, 'Failed')
        count += 1

print(count / len(files))
