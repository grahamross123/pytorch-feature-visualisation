from src.features import view_feature_detectors, view_feature_maps, view_activation_max, view_deconvnet, view_network_inversion, view_grad_cam
from src.load import load_image, load_model_alexnet, load_model
from src.util import imshow
from src.model import AlexNet, CNN

IMAGE_PATH = "./test/1700.jpg"
MODEL_PATH = "./cats_dogs_model.pt"

def main():
    net = AlexNet()
    load_model_alexnet(net)
    # net = CNN()
    # net = load_model(net, MODEL_PATH)

    img, raw_img = load_image(IMAGE_PATH, imsize=244, grayscale=True)
    # view_feature_detectors(net, num=6)
    # imshow("Input image", raw_img)
    # view_feature_maps(net, img, num=8)
    # view_activation_max(net, layer=0, neuron=10, steps=100, lr=0.1, size=100, upscaling_steps=8, upscaling_factor=1.2)
    # view_deconvnet(net, img, num=5)
    # view_network_inversion(net, img)
    view_grad_cam(net, img, raw_img, layer=4)


if __name__ == "__main__":
    main()

