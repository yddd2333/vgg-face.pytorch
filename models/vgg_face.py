import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import torch.utils.data
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
import json
from vgg_face_dag import vgg_face_dag

parser = argparse.ArgumentParser(description='Pytorch')
parser.add_argument('--batchsize', default=6, type=int)
args = parser.parse_args()

# class VGG_16(nn.Module):
#     """
#     Main Class
#     """
#
#     def __init__(self):
#         """
#         Constructor
#         """
#         super().__init__()
#         self.block_size = [2, 2, 3, 3, 3]
#         self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
#         self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
#         self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
#         self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
#         self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
#         self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.fc6 = nn.Linear(512 * 7 * 7, 4096)
#         self.fc7 = nn.Linear(4096, 4096)
#         self.fc8 = nn.Linear(4096, 2622)
#
#     def load_weights(self, path="../pretrained/VGG_FACE.t7"):
#         """ Function to load luatorch pretrained
#
#         Args:
#             path: path for the luatorch pretrained
#         """
#         model = torchfile.load(path)
#         counter = 1
#         block = 1
#         for i, layer in enumerate(model.modules):
#             if layer.weight is not None:
#                 if block <= 5:
#                     self_layer = getattr(self, "conv_%d_%d" % (block, counter))
#                     counter += 1
#                     if counter > self.block_size[block - 1]:
#                         counter = 1
#                         block += 1
#                     self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
#                     self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
#                 else:
#                     self_layer = getattr(self, "fc%d" % (block))
#                     block += 1
#                     self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
#                     self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
#
#     def forward(self, x):
#         """ Pytorch forward
#
#         Args:
#             x: input image (224x224)
#
#         Returns: class logits
#
#         """
#         x = F.relu(self.conv_1_1(x))
#         x = F.relu(self.conv_1_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_2_1(x))
#         x = F.relu(self.conv_2_2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_3_1(x))
#         x = F.relu(self.conv_3_2(x))
#         x = F.relu(self.conv_3_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_4_1(x))
#         x = F.relu(self.conv_4_2(x))
#         x = F.relu(self.conv_4_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv_5_1(x))
#         x = F.relu(self.conv_5_2(x))
#         x = F.relu(self.conv_5_3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc6(x))
#         x = F.dropout(x, 0.5, self.training)
#         x = F.relu(self.fc7(x))
#         x = F.dropout(x, 0.5, self.training)
#         return self.fc8(x)


class myDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.img_path_list = []
        self.mean_param = [129.186279296875/255, 104.76238250732422/255, 93.59396362304688/255]
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.mean_param, std=[1, 1, 1])
    ])

        for img in os.listdir(folder_path):
            img_path = folder_path + img
            self.img_path_list.append(img_path)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        im = Image.open(img_path)
        if im is None:
            return -1, -1
        im = self.transform(im)
        return im.float()*255, img_path.split('/')[-1]

    def __len__(self):
        return len(self.img_path_list)

def get_model():
    print('Loading model...', flush=1)
    # model = VGG_16().double().cuda()
    # model.load_weights()
    model = vgg_face_dag('/home/SENSETIME/dengyang/PycharmProjects/vgg-face'
                         '.pytorch/models/vgg_face_dag.pth').cuda()
    model.eval()

    print('Loading finished')
    return model


# Get prediction result of origin input folder
def folder_pred(folder_path, model):
    dataset = myDataset(folder_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    res = {}
    for batch_idx, (imgs, names) in enumerate(dataloader):
        preds = F.softmax(model(imgs.cuda()), dim=1)
        values, indices = torch.max(preds.data, 1)
        values = values.cpu().detach().numpy()
        indices = indices.cpu().numpy()
        for idx, name in enumerate(names):
            res[name] = (values[idx], indices[idx])
    return res



# Get prediction result of augmentated input folder
def augment_folder_pred(folder_path, model, origin):
    dataset = myDataset(folder_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    res = {}
    for batch_idx, (imgs, names) in enumerate(dataloader):
        preds = F.softmax(model(imgs.cuda()), dim=1)
        preds = preds.cpu().detach().numpy()
        for idx, name in enumerate(names):
            ori_value, label = origin[name]
            value = preds[idx][label]
            dif = ori_value - value
            res[name] = (value, label, dif)
    return res



def vggData_pred(model, path, origin_result=None):
    print('===>' + path.split('/')[-2], flush=1)
    folder_list = os.listdir(path)
    res = {}
    if origin_result is None:
        for idx, folder in enumerate(folder_list):
            print('%d/%d | Processing: ' %(idx+1, len(folder_list)) + folder, flush=1)
            res[folder] = folder_pred(path + folder + '/', model)
    else:
        for idx, folder in enumerate(folder_list):
            print('%d/%d | Processing: ' %(idx+1, len(folder_list)) + folder, flush=1)
            res[folder] = augment_folder_pred(path + folder + '/', model, origin_result[folder])

    return res

def save(dictObj, name):
    jsObj = json.dumps(dictObj)
    fileObject = open(name, 'w')
    fileObject.write(jsObj)
    fileObject.close()


def main():
    # Get pretrained VGG-Face model
    model = get_model()

    pred_origin_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/new_images/')
    pred_mosaic_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/mosaic_images/', origin_result=pred_origin_result)
    pred_block_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/block_images/', origin_result=pred_origin_result)
    pred_blur_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/blur_images/', origin_result=pred_origin_result)



if __name__ == "__main__":
    model = get_model()

    pred_origin_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/new_images/')
    pred_mosaic_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/mosaic_images/', origin_result=pred_origin_result)
    pred_block_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/block_images/', origin_result=pred_origin_result)
    pred_blur_result = vggData_pred(model, '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/blur_images/', origin_result=pred_origin_result)