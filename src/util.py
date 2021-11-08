import matplotlib.pyplot as plt
import math
import random
from torchvision import transforms


def imshow(title, img):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_grid(data, title=None, num=0, shuffle=False):
    """
    Given a 3d tensor of neurons in a convolutional layer, plot these
    in a grid.
    """
    if num > data.size(0):
        raise Exception("num cannot be larger than total number of maps in layer")

    if not num:
        num = data.size(0)

    x = math.floor(num**(1/2))
    y = num**(1/2)
    if not y.is_integer():
        y += 2
    y = int(y)
    fig, ax = plt.subplots(x, y)
    if title:
        fig.suptitle(title)
    h = 0
    v = 0

    if num and shuffle:
        int_range = random.sample(range(data.size(0)), num)
    else:
        int_range = range(num)

    for i in int_range:
        if h == math.floor(num ** (1/2)):
            v += 1
            h = 0
        ax[h, v].imshow(data[i])
        ax[h, v].axes.xaxis.set_visible(False)
        ax[h, v].axes.yaxis.set_visible(False)
        h += 1

    plt.show()


def normalise(img):
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalise(img)

def denormalise(img):
    denormalise = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    return denormalise(img)


def convert_to_plottable(img):
    img = denormalise(img.clone().detach()).numpy()[0]
    # img = img[0] # If not RGB
    img = img.transpose(1, 2, 0) # If RGB
    img = img.clip(0, 1)
    return img


