import os
import PIL.Image
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from utils import dataset_imagenetvalpart
import argparse


def get_bw_hook(save_path, img_filename):
    """
    Summary:
        Defines the function with the right signature to be created for the backward hook.
    """
    def bw_hook(module, grad_input, grad_output):
        return get_grad_norm_hook(module, grad_input, grad_output, save_path, img_filename)
    
    # return the hook function as if it were a string
    return bw_hook


def get_grad_norm_hook(module, grad_input, grad_output, save_path, img_filename):
    '''
    Summary:
        Save l2-norms for each channel.
    '''
    grad = grad_output[0].detach()
    channel_grad_norm = torch.norm(grad, dim=(2,3)).squeeze(0)
    # print(torch.mean(channel_grad_norm))
    file_stub = os.path.basename(img_filename).split('.')[0]
    np.save(os.path.join(save_path, '{}.npy'.format(file_stub)), channel_grad_norm.clone().to('cpu').numpy())


def save_stats(model_dict, dataset, data_transforms, num_images, num_conv_layers, print_every, save_path):
    """
    Summary:
        Save l2-norms for each channel for each layer for each model.
    """
    print("Saving l2-norms to {}...".format(save_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    modules_dict = {}

    for j in model_dict.keys():
        # get required modules
        model = model_dict[j]
        modules_dict[j] = []
        layer_count = 0
        for ind, (nm, module) in enumerate(model.named_modules()):
            # print(ind, nm)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                layer_count += 1
                modules_dict[j].append(module)
                if layer_count == num_conv_layers:
                    break

    os.makedirs(save_path, exist_ok=True)

    image_count = 0
    for i in dataset:
        image_count += 1
        img = i['image']
        filename = i['filename']

        img = img.to(device)
        img = img.unsqueeze(0)

        for j in model_dict.keys():
            handles = []

            for k in range(num_conv_layers):
                final_path = os.path.join(save_path, os.path.join('{}_convlayer'.format(k), j))
                os.makedirs(final_path, exist_ok=True)
                h = modules_dict[j][k].register_backward_hook(get_bw_hook(final_path, filename))
                handles.append(h)

            model = model_dict[j]
            model.to(device)
            model.zero_grad()

            out = model(img)

            # use most interesting class as label (i.e. class with highest softmax probability)
            with torch.no_grad():
                label = torch.max(out, 1)[1]
                label = label.to(device)

            loss = criterion(out, label)
            loss.backward()
            # torch.softmax(out, dim=-1).max().backward()

            for h in handles:
                h.remove()

        if image_count % print_every == 0:
            print("Saved l2-norms for {}/{} images.".format(image_count, num_images))

        if image_count == num_images:
            break


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load pretrained VGG models
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    vgg16_bn = models.vgg16_bn(pretrained=True)
    # to let batch normalisation work for vgg16_bn
    vgg16_bn.eval()
    model_dict = {'vgg16': vgg16, 'vgg16_bn': vgg16_bn}

    num_conv_layers = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='direct parent folder path containing the imgnet500 images')
    parser.add_argument('--save_norms_folder', type=str, help='folder path to save the l2-norms')
    parser.add_argument('--print_every', default=10, type=int, help='number of images before printing progress')
    parser.add_argument('--num_images', default=250, type=int, help='number of images to save l2-norms for')
    args = parser.parse_args()

    dataset = dataset_imagenetvalpart(root_dir=args.img_path, maxnum=0, transform=data_transforms)

    save_stats(model_dict, dataset, data_transforms, args.num_images, num_conv_layers, args.print_every, args.save_norms_folder)