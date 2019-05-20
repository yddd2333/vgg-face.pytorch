import os
import cv2

folder_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/images/'
for subFolder in os.listdir(folder_path):
    subFolder_path = folder_path + subFolder + '/'
    print('cleaning ' + subFolder, flush=1)
    for img in os.listdir(subFolder_path):
        img_path = subFolder_path + img
        im = cv2.imread(img_path)
        if im is None or os.path.getsize(img_path)<2000:
            os.remove(img_path)