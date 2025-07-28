import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from Config.config import get_config_dict
from dataset.dataset import vehicledata


def get_dataloader():
    val_dataset = vehicledata(cfg['dataset']['test_path'], cfg['dataset']['test_ann_path'],
                              cfg['dataset']['num_class'], cfg['dataset']['size'])

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['args']['batch_size'], shuffle=True,
                                         num_workers= cfg['args']['num_workers'])
    return loader

def pred_to_rgb(pred):
    assert len(pred.shape) == 3
    #
    pred = pred.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    pred = pred.detach().cpu().numpy()
    #
    pred_rgb = np.zeros_like(pred, dtype=np.uint8)
    pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(0, 21):
        pred_rgb[pred == i] = np.array(color_table[i])

    plt.imshow(pred_rgb)

    return pred_rgb

def trg_to_rgb(target):
    assert len(target.shape) == 3
    #
    target = target.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
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
    for i in range(21):
        target_rgb[target == i] = np.array(color_table[i])

    plt.imshow(target_rgb)

    return target_rgb


def matplotlib_imshow(img):

    npimg = img.numpy()

    npimg = (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)
    plt.imshow(npimg)


if __name__=='__main__':
    cfg = get_config_dict()


    loader = get_dataloader()

    org_path = '/storage/sjpark/vehicle_data/checkpoints/new_dataloader/DeepLab/256/Original_DeepLabV3+/Original_ResNet50_DeepLabV3+.pth'
    repblock_path = '/storage/sjpark/vehicle_data/checkpoints/new_dataloader/DeepLab/256/RepVGG_DeepLabV3+/RepVGG_ResNet50_75456.pth'
    attention_path = '/storage/sjpark/vehicle_data/checkpoints/new_dataloader/DeepLab/256/Apply_DA(channel)_ECA/bottleneck1+bottleneck2+bottleneck3/ResNet50_DeepLabV3+_75456_DA_ECA_bottleneck1+bottleneck2+bottleneck3.pth'

    from model.deeplabv3plus import *
    org_model = DeepLab(num_classes=cfg['dataset']['num_class'], backbone=cfg['solver']['backbone'],
                        output_stride=cfg['solver']['output_stride'], sync_bn=False, freeze_bn=False, pretrained=False)

    ckpt = torch.load(org_path, map_location='cpu')
    resume_state_dict = ckpt['model'].state_dict()
    try:
        org_model.load_state_dict(resume_state_dict, strict=True)
        print("Load_weight")
    except:
        print("Not load_weight")


    from model.RepVGG_ResNet_deeplabv3plus import *
    repblock_model = DeepLab(num_classes=cfg['dataset']['num_class'], backbone=cfg['solver']['backbone'],
                        output_stride=cfg['solver']['output_stride'], sync_bn=False, freeze_bn=False, pretrained=False, deploy=True)

    ckpt = torch.load(repblock_path, map_location='cpu')
    try:
        print("Load_weight")
        repblock_model.load_state_dict(ckpt, strict=True)
    except:
        print("Not load_weight")


    from model.RepVGG_DA_ECA_ResNet_deeplabv3plus import *

    attention_model = DeepLab(num_classes=cfg['dataset']['num_class'], backbone=cfg['solver']['backbone'],
                             output_stride=cfg['solver']['output_stride'], sync_bn=False, freeze_bn=False,
                             pretrained=False, deploy=True)

    ckpt = torch.load(attention_path, map_location='cpu')
    try:
        print("Load_weight")
        attention_model.load_state_dict(ckpt, strict=True)
    except:
        print("Not load_weight")





    for batch_idx, (data, target, label, idx) in enumerate(loader):

        output1 = org_model(data)
        output2 = repblock_model(data)
        output3 = attention_model(data)

        trg_to_rgb(target[0])

        pred_to_rgb(output1[0])
        pred_to_rgb(output2[0])
        pred_to_rgb(output3[0])
