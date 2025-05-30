from model.RepVGG_ResNet_deeplabv3plus import DeepLab
from PIL import Image
from torchvision import transforms

import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']

color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
               5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
               9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
               13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
               17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

def resize_train(image, size = 256):
    resize = transforms.Resize(size)
    return resize(image)

def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1

    return img

def pred_to_rgb(pred):
    assert len(pred.shape) == 3
    #
    pred = pred.softmax(dim=0).argmax(dim=0).to('cpu')
    #
    pred = pred.detach().cpu().numpy()
    #
    pred_rgb = np.zeros_like(pred, dtype=np.uint8)
    pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
    #
    for i in range(len(CLASSES)):
        pred_rgb[pred == i] = np.array(color_table[i])

    return pred_rgb

def add_bounding_box(pred, PT):
    vehicle_class_id = [1, 2, 3, 4, 5, 6, 7]
    #
    pred = torch.softmax(pred, dim=0)
    pred = pred.detach().cpu().numpy()
    #
    bounding_box = []

    for class_id in vehicle_class_id:
        class_prob_map = pred[class_id]
        class_mask = (class_prob_map >= PT[CLASSES[class_id]]).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 30:
                continue
            bounding_box.append((class_id, (x, y, w, h)))

    return bounding_box

if __name__ =='__main__':
    # device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

    image_path = '/storage/sjpark/vehicle_data/2_crop/2_crop'

    train_dir = sorted(os.listdir(image_path))

    color_table_bgr = {k: (v[2], v[1], v[0]) for k, v in color_table.items()}

    PT = {'vehicle': 0.2, 'bus': 0.2, 'truck': 0.2, 'policeCar': 0.2, 'ambulance': 0.2, 'schoolBus': 0.2,
          'otherCar': 0.2}

    model1 = DeepLab(num_classes=21, backbone='resnet50', output_stride=16, sync_bn=False, freeze_bn=False, pretrained=False, deploy=True)
    file_path1 = '/storage/sjpark/vehicle_data/checkpoints/new_dataloader/DeepLab/256/Apply_DA(channel)_ECA/bottleneck1+bottleneck2+bottleneck3/ResNet50_DeepLabV3+_75456_DA_ECA_bottleneck1+bottleneck2+bottleneck3.pth'

    ckpt = torch.load(file_path1, map_location='cpu')
    model1.load_state_dict(ckpt, strict=True)
    model1 = model1.eval()

    # model2 = DeepLab(num_classes=21, backbone='resnet101', output_stride=16, sync_bn=False, freeze_bn=False, pretrained=False, deploy=True)
    # file_path2 = '/storage/sjpark/vehicle_data/checkpoints/new_dataloader/DeepLab/256/Apply_ECA/bottleneck2/ResNet101_DeepLabV3+_75456_ECA_after_bottleneck2.pth'
    #
    # ckpt = torch.load(file_path2, map_location='cpu')
    # model2.load_state_dict(ckpt, strict=True)
    # model2 = model2.eval()

    for idx, data in enumerate(train_dir):
        img = os.path.join(image_path, train_dir[10])
        img = Image.open(img)
        img = resize_train(img, (256, 256))
        img = np.array(img, dtype=np.uint8)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = transform(img).unsqueeze(0)

        out1 = model1(img)
        # out2 = model2(img)

        pred_out1 = pred_to_rgb(out1[0])
        pred_out1 = cv2.cvtColor(pred_out1, cv2.COLOR_RGB2BGR)
        #
        # pred_out2 = pred_to_rgb(out2[0])
        # pred_out2 = cv2.cvtColor(pred_out2, cv2.COLOR_RGB2BGR)

        # pred_out1_bounding_box = add_bounding_box(out1[0], PT)
        # pred_out2_bounding_box = add_bounding_box(out2[0], PT)

        pred_out1 = cv2.addWeighted(image, 1, pred_out1, 0.5, 0)
        # pred_out2 = cv2.addWeighted(image, 1, pred_out2, 0.5, 0)

        # for idx, (x, y, w, h) in pred_out1_bounding_box:
        #     cv2.rectangle(pred_out1, (x, y), (x+w, y+h), color_table_bgr[idx], 1)
        #
        # for idx, (x, y, w, h) in pred_out2_bounding_box:
        #     cv2.rectangle(pred_out2, (x, y), (x+w, y+h), color_table_bgr[idx], 1)



        # separator = np.ones((256, 5, 3), dtype=np.uint8) * 255

        # concat_image = np.concatenate((pred_out1, separator, pred_out2), axis=1)

        concat_image = cv2.resize(pred_out1, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Image", concat_image)
        cv2.waitKey(33)
