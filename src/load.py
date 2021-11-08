import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
from src.model import AlexNet, CNN
import torchvision.transforms.functional as FT


def load_image(filename, imsize, grayscale=False):
    input_image = Image.open(filename)

    crop = transforms.Compose([
        transforms.Resize(imsize + 12),
        transforms.CenterCrop(imsize),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_cropped = crop(input_image)
    input_tensor = preprocess(input_cropped)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch, np.array(input_cropped)


def load_model_alexnet(net):
    mapping = {'features.0.weight': 'deConv1.weight',
               'features.3.weight': 'deConv2.weight',
               'features.6.weight': 'deConv3.weight',
               'features.8.weight': 'deConv4.weight',
               'features.10.weight': 'deConv5.weight'}
    state_dict = load_state_dict_from_url("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth")
    for key, value in state_dict.copy().items():
        if key in mapping:
            state_dict[mapping[key]] = value
    net.load_state_dict(state_dict)


def load_model(net, model_path):
    net.load_state_dict(torch.load(model_path))
    return net.eval()