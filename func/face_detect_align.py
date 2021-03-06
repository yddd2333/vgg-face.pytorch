# coding: utf-8
import cv2
import dlib
import numpy as np
import os


def align_img(input_path, output_path):
    predicter_path = './shape_predictor_68_face_landmarks.dat'
    face_file_path = input_path  # 要使用的图片，图片放在当前文件夹中

    # 导入人脸检测模型
    detector = dlib.get_frontal_face_detector()
    # 导入检测人脸特征点的模型
    sp = dlib.shape_predictor(predicter_path)
    bgr_img = cv2.imread(face_file_path)
    if bgr_img is None:
        print("Sorry, we could not load '{}' as an image".format(face_file_path))
        exit()

    # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # 检测图片中的人脸
    dets = detector(rgb_img, 1)
    # 检测到的人脸数量
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        exit()

    # 识别人脸特征点，并保存下来
    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(sp(rgb_img, det))

    # 人脸对齐
    image = dlib.get_face_chips(rgb_img, faces, size=350)[0]

    # 显示对齐结果
    cv_rgb_image = np.array(image).astype(np.uint8)  # 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr
    cv2.imwrite(output_path, cv_bgr_image)
    # cv2.imshow('%s'%(image_cnt), cv_bgr_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def align_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    img_list = os.listdir(input_folder)
    for img in img_list:
        img_path = input_folder + img
        new_img_path = output_folder + img
        align_img(img_path, new_img_path)


def align_dataset(input_folder, output_folder):
    folder_list = os.listdir(input_folder)
    for folder in fodler_list:
        folder_path = input_folder + folder + '/'



if __name__ == '__main__':
    align_folder('/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/func/Anton_Yelchin/350/',
                 '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/func/Anton_Yelchin/resize/')

