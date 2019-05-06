import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

class SpatialTransform(object):
    def __init__(self, is_normalized=True, resize_size=256, crop_size=224):
        super(SpatialTransform, self).__init__()

        self.resized = torchvision.transforms.Resize(resize_size)
        self.crop = torchvision.transforms.FiveCrop(crop_size)
        self.to_tensor = torchvision.transforms.ToTensor()
        if is_normalized:
            self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.normalize = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

    def __call__(self, img):
        
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        imgs = self.crop(self.resized(img))
        imgs = [self.normalize(self.to_tensor(im)) for im in imgs]
        return torch.stack(imgs, dim=0)