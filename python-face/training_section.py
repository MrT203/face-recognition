import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Tạo đường dẫn tới thư mục chứa ảnh
path = 'dataSet'

def getImageWithId(path, id):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')

        faceNp = np.array(faceImg, 'uint8')

        # Extract ID from folder name
        Id = int(id)

        faces.append(faceNp)
        IDs.append(Id)

        cv2.imshow('training', faceNp)
        cv2.waitKey(10)

    return faces, IDs

# Lấy danh sách các thư mục con trong thư mục dataSet
for root, dirs, files in os.walk(path):
    for folder_name in dirs:
        folder_path = os.path.join(root, folder_name)
        faces, Ids = getImageWithId(folder_path, folder_name)
        recognizer.update(faces, np.array(Ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()