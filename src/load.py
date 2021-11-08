import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url


class ImageLoader():
    def __init__(self, img_size):
        self.img_size = img_size
    TRAINING = "./train"
    TESTING = "./test"
    training_data = []
    LABELS = {"cat": 0, "dog": 1}
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for f in tqdm(os.listdir(self.TRAINING)):
            if "jpg" in f:
                if "dog" in f:
                    label = "dog"
                if "cat" in f:
                    label = "cat"
                try:
                    path = os.path.join(self.TRAINING, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    
                    if label == "cat":
                        self.catcount += 1
                    elif label == "dog":
                        self.dogcount += 1

                except Exception as e:
                    print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats:',self.catcount)
        print('Dogs:', self.dogcount)


    def load_data(self):
        self.training_data = np.load("training_data.npy", allow_pickle=True)
        return self.training_data


    def separate_data(self, val_pct):
        X = torch.Tensor([i[0] for i in self.training_data]).view(-1, self.img_size, self.img_size)
        X = X/255
        y = torch.Tensor([i[1] for i in self.training_data])
        val_size = int(len(X)*val_pct)
        return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]


def load_image(filename):
    input_image = Image.open(filename)

    crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_cropped = crop(input_image)
    input_tensor = preprocess(input_cropped)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch, np.array(input_cropped)


def load_model(net):
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