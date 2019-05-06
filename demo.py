import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import glob
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from spatial_transforms import Threshold
if __name__ == '__main__':
    label2index = {}
    label2index[0] = 'fire'
    label2index[1] = 'normal'
    label2index[2] = 'smoke'
    label2index['fire'] = 0
    label2index['normal'] = 1 
    label2index['smoke'] = 2
    
    opt = parse_opts()
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    
    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    
    spatial_transform = transforms.Compose([
        transforms.Resize(256),#(224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #Threshold(200),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    torch.set_grad_enabled(False)
    model.eval()
    
    print('run')

    test_images = glob.glob('test/*.jpg')
    test_images = sorted(test_images)
    for image in test_images:
        img = Image.open(image)
        img = spatial_transform(img)
        img = torch.unsqueeze(img, 0)
        # print(img.size())
        output = model(img)
        output = F.softmax(output, dim = -1)
        val, ind = torch.topk(output, 1)
        #print(image, output, val.item(), label2index[ind.item()])
        if label2index[ind.item()] in image :
            flag = ''
        else:
            flag = 'error'
        print(image, val.item(), label2index[ind.item()], flag)

