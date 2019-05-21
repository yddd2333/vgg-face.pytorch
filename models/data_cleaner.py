import os
import cv2
import augmentor



if __name__ == '__main__':
    folder_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/images/'
    new_folder_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/RGBimages/'
    subFolders = os.listdir(folder_path)
    numFolders = len(subFolders)
    part = 0.25
    for i, subFolder in enumerate(subFolders[int((part-0.05)*numFolders):int(part*numFolders)]):
        subFolder_path = folder_path + subFolder + '/'
        newSubFolder_path = new_folder_path+subFolder+'/'
        print('%d:%d-%d'%(i+int((part-0.05)*numFolders), int((part-0.05)*numFolders), int(part*numFolders)), flush=1)
        for img in os.listdir(subFolder_path):
            img_path = subFolder_path + img
            im = cv2.imread(img_path)
            if im is None or os.path.getsize(img_path)<2000:
                os.remove(img_path)
        if not os.path.exists(newSubFolder_path):
            os.makedirs(newSubFolder_path)
        augmentor.augment_folder(subFolder_path, newSubFolder_path, augmentor.RGB)
