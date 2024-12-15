import os
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']


def resize_train(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


def resize_label(image, size):
    resize_img = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=size, mode='nearest').squeeze()
    return resize_img


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
        img_orig = np.array(img, dtype=np.uint8)

        # Label
        label = os.path.join(self.annotation_path + self.ann_file[index])
        label = Image.open(label)
        label_orig = np.array(label, dtype=np.uint8)

        img, label = self.transform(img_orig, label_orig)

        img = resize_train(img, self.size)
        label = resize_label(label, self.size)

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        # Debugging


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


def trg_to_rgb(target):
    assert len(target.shape) == 3
    #
    target = target.softmax(dim=0).argmax(dim=0).to('cpu')
    #
    target = target.detach().cpu().numpy()
    #
    target_rgb = np.zeros_like(target, dtype=np.uint8)
    target_rgb = np.repeat(np.expand_dims(target_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(len(CLASSES)):
        target_rgb[target == i] = np.array(color_table[i])

    return target_rgb

if __name__ == "__main__":
    image_path = "/storage/sjpark/vehicle_data/Dataset/train_image/"
    annotation_path = "/storage/sjpark/vehicle_data/Dataset/ann_train/"
    dataset_object = vehicledata(image_path, annotation_path, 21, (256, 256))

    img, target, label, index = dataset_object.__getitem__(0)
    import matplotlib.pyplot as plt