import os
from idlelib.pyparse import trans

import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resize_train(image, size):
    resize = transforms.Resize(size)
    return resize(image)


def resize_label(image, size):
    resize = transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    return resize(image)


class vehicledata():
    CLASSES = (
        'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
        'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
        'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
        'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
    )

    def __init__(self, image_path, annotation_path, n_class, size, transform=None):
        self.image_path = image_path
        self.train_dir = sorted(os.listdir(self.image_path))
        #
        self.annotation_path = annotation_path
        self.ann_file = sorted(os.listdir(self.annotation_path))
        #
        self.size = size
        self.n_class = n_class

    def __len__(self):
        return len(self.train_dir)

    def __getitem__(self, index):
        #
        assert self.train_dir[index].split('.')[0] == self.ann_file[index].split('.')[0], f'file names are different...'

        # Training_image
        img = os.path.join(self.image_path, self.train_dir[index])
        img = Image.open(img)

        # Label
        label = os.path.join(self.annotation_path + self.ann_file[index])
        label = Image.open(label)

        img = resize_train(img, self.size)
        label = resize_label(label, self.size)

        img = np.array(img, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)
        img, label = self.transform(img, label)


        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1


        return img, target, label, index

    def transform(self, img, label):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1
        label = torch.from_numpy(label).float()

        return img, label

    def untransform(self, img, label):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] * 255.0
        label = label.numpy()

        return img, label


if __name__ == "__main__":
    image_path = "/storage/sjpark/vehicle_data/Dataset/train_image/"
    annotation_path = "/storage/sjpark/vehicle_data/Dataset/ann_train/"
    dataset_object = vehicledata(image_path, annotation_path, 21, (256, 256))

    img, target, label,  index = dataset_object.__getitem__(0)
