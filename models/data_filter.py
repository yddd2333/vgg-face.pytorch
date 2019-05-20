# -*- coding: utf-8 -*-

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchfile
import numpy as np
import os
import shutil
import numpy as np

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

class myDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.img_path_list = []
        for img in os.listdir(folder_path):
            img_path = folder_path + img
            self.img_path_list.append(img_path)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        im = cv2.imread(img_path)
        if im is None:
            return -1, -1
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
        im = torch.Tensor(im).permute(2, 0, 1).view(3, 224, 224).double()
        im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(3, 1, 1)
        return im, img_path.split('/')[-1]

    def __len__(self):
        return len(self.img_path_list)


def folder_filter(folder_path, new_folder_path, model):
    print('Processing: ' + folder_path.split('/')[-2], flush=1)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    dataset = myDataset(folder_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=8, shuffle=False, num_workers=2)
    for i, (img, name) in enumerate(dataloader):
        preds = F.softmax(model(img.cuda()), dim=1)
        values, indices = torch.max(preds.data, 1)
        values = values.cpu().detach().numpy()
        ind = np.where(values>0.5)[0]
        for j in ind:
            shutil.copy(folder_path + name[j], new_folder_path + name[j])



# Return the result of prediction
'''def get_pred(img_path, model):
    im = cv2.imread(img_path)
    if im is None:
        return -1, -1
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double().cuda()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1).cuda()
    preds = F.softmax(model(im), dim=1)
    values, indices = preds.max(-1)
    return int(indices.cpu().numpy()[0]), float(values.cpu().detach().numpy()[0])


def folder_filter(folder_path, new_folder_path, model):
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    img_list = os.listdir(folder_path)
    print('Processing: ' + folder_path.split('/')[-2] + ' | Number of images:' +str(len(img_list)), flush=1)
    for i, img in enumerate(img_list):
        ind, val = get_pred(folder_path + img, model)
        if val>0.5:
            shutil.copy(folder_path + img, new_folder_path + img)'''


def vggData_filter(model):
    path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/images/'
    new_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/new_images/'
    folder_list = os.listdir(path)
    for i, folder in enumerate(folder_list):
        if i%10 == 0:
            print('%d/%d' %(i, len(folder_list)))
        folder_filter(path + folder + '/', new_path + folder + '/', model)



if __name__ == "__main__":
    print('Loading model...', flush=1)
    model = VGG_16().double().cuda()
    model.load_weights()
    model.eval()
    print('Loading finished', flush=1)
    # folder_filter('/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/images/A.R._Rahman/',
    #               '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/new_images/0.2_0.5/',
    #               model)
    vggData_filter(model)