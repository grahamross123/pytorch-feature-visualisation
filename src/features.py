from numpy.typing import _256Bit
import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
from src.util import imshow, plot_grid, normalise, convert_to_plottable, denormalise
import matplotlib.pyplot as plt
from torchvision import transforms
from copy import copy
import numpy as np
import cv2

TOTAL_CONV_LAYERS = 5
CONV_TO_LAYER = {0: 0, 1: 3, 2: 6, 3: 8, 4: 10} # Converts a given conv layer number to a total layer number in the network

def view_feature_detectors(net, num=0):
    """
    Plot the feature detectors of a network.
    """
    print("Viewing feature detectors...")
    for layer in net.children():
        if isinstance(layer, nn.Sequential):
            for layer in layer.children():
                if isinstance(layer, nn.Conv2d):
                    weights = layer.state_dict()["weight"]
                    f_min, f_max = weights.min(), weights.max()
                    filters = (weights - f_min) / (f_max - f_min)

                    if filters.shape[1] == 3:
                        filt_images = filters.transpose(1, 3)

                    else:
                        filt_images = filters[:, 0, :,:]
                    plot_grid(filt_images, title=layer, num=num)
        break # Only view filters in first sequential layer


def view_feature_maps(net, input_img, num=8):
    """
    View the feature maps for all conv layers of a network given an input image
    """
    print("Viewing feature maps...")
    # Run the input image through the network to obtain the feature outputs
    net(input_img)
    for layer, filter_map in net.feature_outputs.items():
        filter_map = filter_map[0, :, :].detach()
        plot_grid(filter_map, title=net.layer_names[layer], num=num)


def activation_max(net, layer=0, neuron=0, steps=100, lr=0.1, size=100, upscaling_steps=1, upscaling_factor=1.2):
    """
    Use activation maximisation to iteratively find the input image that maximises
    the activation of a given neuron.
    """
    layer_num = layer
    # input_image = torch.randn(3, 224, 224).view(1, 3, 224, 224)
    img = np.single(np.random.uniform(0,1, (3, size, size)))
    img = normalise(torch.from_numpy(img)).view(1, 3, size, size)
    net(img)
    net.requires_grad_(False)

    def act_max_loss(activation, img):
        pxl_inty = torch.pow((img**2).mean(), 0.5)
        rms = torch.pow((activation**2).mean(), 0.5)
        return -rms #+ pxl_inty

    loss = 0 # Set loss to 0 for initial log
    print("Activation Max: Optimising image...")
    for i in range(upscaling_steps):
        print(f"Upscaling step {i+1} of {upscaling_steps} | resolution {size} x {size} | loss = {round(loss.item(), 3) if loss else 0}")
        img_var = copy(img)
        img_var.requires_grad_(True)
        optimizer = optim.Adam([img_var], lr=lr)
        for i in tqdm(range(steps)):
            net(img_var)
            optimizer.zero_grad()
            if neuron == -1:
                loss = act_max_loss(net.feature_outputs[layer_num][0, :, :, :], img_var)
            else:
                loss = act_max_loss(net.feature_outputs[layer_num][0, neuron, :, :], img_var)
            loss.backward()
            optimizer.step()
            img = regularise_image(img)
            
        size = int(upscaling_factor * size)  # calculate new image size
        img = transforms.functional.resize(img_var, (size, size))
    return img

def regularise_image(img):
    jitter = transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0)
    rotation = transforms.RandomRotation(degrees=(-5,5))
    blur = transforms.GaussianBlur(kernel_size=5, sigma=0.1)
    img = jitter(img)
    img = rotation(img)
    img = blur(img)
    return img

def view_activation_max(net, layer=0, neuron=0, steps=100, lr=0.1, size=100, upscaling_steps=1, upscaling_factor=1.2):
    img = activation_max(net, layer=layer, neuron=neuron, steps=steps, lr=lr, size=size, upscaling_steps=upscaling_steps, upscaling_factor=upscaling_factor)
    imshow(f"Layer {layer}, neuron {neuron}", convert_to_plottable(img))


def view_deconvnet(net, input_img, num=10):
    """
    Plots the outputs from the deconvnet for a given number of neurons in each layer.
    """
    print("Viewing deconvnet...")
    fig, ax = plt.subplots(num, TOTAL_CONV_LAYERS)
    for layer_num in range(TOTAL_CONV_LAYERS):
        for neuron_num in range(num):
            if neuron_num == 0:
                ax[neuron_num, layer_num - 1].set_title(f"layer {layer_num}")
            output = deconv_neuron(net, input_img, layer_num, neuron_num)
            ax[neuron_num, layer_num].axes.xaxis.set_visible(False)
            ax[neuron_num, layer_num].axes.yaxis.set_visible(False)
            ax[neuron_num, layer_num].imshow(output)
    fig.suptitle("Deconvnet")
    plt.tight_layout()
    plt.show()


def deconv_neuron(net, input_img, conv_layer=0, neuron=0):
    """
    Takes the activation of a neuron and runs it through the deconvnet to output 
    an image in the original image dimension.
    """
    if conv_layer > 4 or conv_layer < 0:
        raise Exception(f"conv_layer should be between 1 and 5, not {conv_layer}")
    net(input_img)

    # 1: obtain feature map from given layer / neuron
    features = net.feature_outputs[CONV_TO_LAYER[conv_layer]]
    act_map = features[0, neuron, :, :]
    output = torch.zeros(features.shape)
    output[0, neuron, :, :] = act_map

    # 2: run output from given feature map through the correct deconv network
    # output_image_dim = net.forward_deconv(output, conv_layer)
    output_image_dim = net.forward_deconv(net.feature_outputs[CONV_TO_LAYER[conv_layer]], conv_layer)

    # 3: return output in image dimension
    return convert_to_plottable(output_image_dim)


def network_inversion(net, img_input, lr=0.1, steps=500, layer_num=0):
    """
    Iteratively find an image that closest fits the activation maps of a given layer
    for an input image.
    """
    img_var = np.single(np.random.uniform(0,1, (3, 224, 224)))
    img_var = normalise(torch.from_numpy(img_var)).view(1, 3, 224, 224)
    net(img_input)

    def loss_fn(img_var, img_input):
        net(img_input)
        feature_var = net.feature_outputs[layer_num][0] # Feature map after input of variable image
        net(img_var)
        feature_input = net.feature_outputs[layer_num][0] # Target feature map
        loss = nn.MSELoss()

        return loss(feature_var, feature_input)

    net.requires_grad_(False)
    img_var.requires_grad_(True)
    optimizer = optim.Adam([img_var], lr=lr, weight_decay=1e-6)
    for _ in tqdm(range(steps)):
        optimizer.zero_grad()
        loss = loss_fn(img_var, img_input)
        loss.backward()
        optimizer.step()
    return convert_to_plottable(img_var)


def view_network_inversion(net, img_input, lr=0.1, steps=100):
    """
    View the images from the network inversion algorithm for each convolutional layer.
    """
    fig, ax = plt.subplots(1, 5)
    print(f"Network inversion: Optimising image...")
    fig.suptitle("Network inversion")
    for i, conv_layer in enumerate(range(TOTAL_CONV_LAYERS)):
        print(f"Layer {conv_layer} of {TOTAL_CONV_LAYERS}")
        img = network_inversion(net, img_input, steps=steps, lr=lr, layer_num=CONV_TO_LAYER[conv_layer])
        ax[i].imshow(img)
        ax[conv_layer].axes.xaxis.set_visible(False)
        ax[conv_layer].axes.yaxis.set_visible(False)
        ax[conv_layer].set_title(f"CL{conv_layer+1}")
    plt.show()


def grad_cam(cnn, img):
    pred = cnn(img)
    print(pred.shape)
    print(pred[:, pred.argmax(dim=1)].shape)
    pred[:, pred.argmax(dim=1)].backward()
    gradients = cnn.gradients
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = cnn.feature_outputs[11]
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    print(activations.shape)
    heatmap = torch.mean(activations, dim=1).squeeze().detach()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    print(heatmap.shape)
    return heatmap.detach().numpy()


def view_grad_cam(cnn, img, raw_img):
    heatmap = grad_cam(cnn, img)
    # imshow("grad cam heatmap",heatmap)
    heatmap = cv2.resize(heatmap, (raw_img.shape[0], raw_img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    plt.imshow(raw_img)
    plt.imshow(heatmap, alpha=0.5, cmap="jet")
    plt.show()


def get_all_activation(net, input_img):
    """
    Retrieve a dictionary with the filter maps for each conv layer of the network
    for a specific input image using hooks.
    """
    activation = {}
    def hook(model, input, output):
        activation[model] = output.detach()

    def get_all_layers(net):
        for name, layer in net._modules.items():
            #If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                get_all_layers(layer)
            elif isinstance(layer, nn.Conv2d):
                print(layer)
                # it's a non sequential. Register a hook
                layer.register_forward_hook(hook)
    get_all_layers(net)
    net(input_img)
    return activation