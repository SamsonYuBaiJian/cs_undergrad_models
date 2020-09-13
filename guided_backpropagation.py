import torch
from utils import dataset_imagenetvalpart
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import argparse


class MyReLUFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        # zero out negative gradients
        grad_input[grad_input < 0] = 0
        return grad_input


class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyReLUFunction.apply(input)


def setbyname(model, name, value):
    """
    Summary:
        Replaces a certain component in a model with value.
    """
    def iteratset(obj, components, value):
        # print('components', components)
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(model, components, value)
    return success


def imshow2(hm, imgtensor, q=100):
    """
    Summary:
        Show image with heatmap.
    """
    def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # print(ten.shape)
        s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
        m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

        res=ten*s+m
        return res

    fig, axs = plt.subplots(1, 2)

    hm = hm.squeeze().sum(dim=0).numpy()

    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim
    axs[1].imshow(hm, cmap="seismic", clim=(-1, 1))
    axs[1].axis('off')

    ts=invert_normalize(imgtensor.squeeze(0))
    a=ts.data.numpy().transpose((1, 2, 0))
    axs[0].imshow(a)
    axs[0].axis('off')

    plt.show()


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='direct parent folder path containing the imgnet500 images')
    parser.add_argument('--model_type', default='vgg16', choices=['vgg16', 'vgg16_bn'], help='model to use')
    parser.add_argument('--show_img_num', type=int, default=1, help='number of images to show, can be left blank to show 1 image')
    args = parser.parse_args()

    # replace ReLU with custom ReLU function
    if args.model_type == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.model_type == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    # to let batch normalisation work for vgg16_bn
    model.eval()

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.ReLU):
            myrelu = MyReLU()
            setbyname(model, n, myrelu)

    # load dataset
    dataset = dataset_imagenetvalpart(root_dir=args.img_path, maxnum=0, transform=data_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # visualise images
    show_img_num = args.show_img_num

    image_count = 0

    for i in dataset:
        img = i['image']
        img = img.to(device)
        img = img.unsqueeze(0)
        img.requires_grad = True

        model.to(device)
        model.zero_grad()
        out = model(img)

        # use most interesting class as label (i.e. class with highest softmax probability)
        label = torch.max(out, 1)[1]
        label = label.to(device)

        loss = criterion(out, label)
        loss.backward()
        # torch.softmax(out, dim=-1).max().backward()

        grad = img.grad.data.to('cpu')
        img = img.to('cpu')
        imshow2(grad, img)
        
        image_count += 1
        if image_count == show_img_num:
            break