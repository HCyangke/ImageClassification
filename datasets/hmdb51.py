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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'frame{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels():
    labels_dic={}
    index=0
    with open('/data/users/yangke/yangke-data/dataset/hmdb51/hmdb_label.csv','r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            labels_dic[line[0]]=index
            index+=1
    print(labels_dic)
    return labels_dic


def get_video_names_and_annotations(subset):
    video_names = []
    annotations = []
    n_frames = []
    file_path="/data/users/yangke/yangke-data/dataset/hmdb51/hmdb_{}_list01.csv".format(subset)
    with open(file_path,'r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            if 'laugh'!=line[1]:
            #if line[2] != 'best_laugh_attack_ever!_laugh_h_nm_np1_fr_goo_2':
                continue
            for i in range(8):
                video_names.append(line[2])
                annotations.append(line[1])
                n_frames.append(int(line[3]))
    return video_names, annotations, n_frames


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    video_names, annotations, n_frames= get_video_names_and_annotations(subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join('/data/users/yangke/yangke-data/dataset/hmdb51/jpegs_256', 
                                video_names[i])
        
        # n_frames_file_path = os.path.join(video_path, 'n_frames')
        # n_frames = int(load_value_file(n_frames_file_path))
        video_frames = n_frames[i]
        if video_frames <= 0:
            continue

        begin_t = 1
        end_t = video_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': video_frames,
            'video_id': video_names[i]
        }
        #video视频帧所在文件夹的位置
        #segment
        #帧的数量
        #视频的名字
        sample['label'] = class_to_idx[annotations[i]]

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, video_frames + 1))
            dataset.append(sample)
        elif n_samples_for_each_video == -1:
            if video_frames<sample_duration:
                sample['frame_indices'] = list(range(1, video_frames + 1))
                dataset.append(sample)
            elif video_frames==sample_duration:
                for j in range(1, 3):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(
                        range(j, min(video_frames + 1, j + sample_duration)))
                    dataset.append(sample_j)
            else:
                #print(video_frames)
                for j in range(1, video_frames - sample_duration + 2):
                    #print(j)
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(
                        range(j, min(video_frames + 1, j + sample_duration)))
                    dataset.append(sample_j)
                ##input()
        else:

            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((video_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, video_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(video_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class HMDB51(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=64,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
