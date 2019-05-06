#coding=utf-8
import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import csv
from utils import load_value_file
import cv2
def get_class_labels():
    labels_dic={}
    labels_dic['fire'] = 0
    labels_dic['normal'] = 1
    #labels_dic['smoke'] = 2
    return labels_dic

def get_video_names_and_annotations(subset):
    images_names = []
    annotations = []
    images_path = []
    file_path="/nfs/private/workspace/clean/data/{}.csv".format(subset)
    with open(file_path,'r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            #if line[2]!='/nfs/project/surveillance/smoke_fire_detection/fire/image/posVideo6_00110.jpg':
            #    continue
            #if line[2]!='v_Basketball_g03_c02':
            #    continue
            #for i in range(8):
            images_names.append(line[1])
            annotations.append(line[0])
            images_path.append(line[2])
    return images_names, annotations, images_path


def make_dataset(subset):
    images_names, annotations, images_path= get_video_names_and_annotations(subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(images_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(images_names)))

        # images_path = images_path[i]
        sample = {
            'image': images_path[i],
            'image_id': images_names[i]
        }
        sample['label'] = class_to_idx[annotations[i]]

        dataset.append(sample)

    return dataset, idx_to_class


class custom(data.Dataset):

    def __init__(self, subset, spatial_transform=None, target_transform=None):
        self.data, self.class_names = make_dataset(subset)

        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        path = self.data[index]['image']

        img = Image.open(path).convert("RGB")
        #tmp = cv2.imread(path)
        #print(path)
        if self.spatial_transform is not None:
            img = self.spatial_transform(img)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
