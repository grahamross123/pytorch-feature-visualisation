from src.features import view_feature_detectors, view_feature_maps, view_activation_max, view_deconvnet, view_network_inversion, view_grad_cam
from src.model import AlexNet, CNN
from src.load import load_image, load_model
from src.util import imshow

def main():
    cnn = AlexNet()
    
    load_model(cnn)
    cnn.eval()
    img, raw_img = load_image("./test/1700.jpg")

    # view_feature_detectors(cnn, num=64)
    # imshow("Input image", raw_img)
    # view_feature_maps(cnn, img, num=8)
    # view_activation_max(cnn, layer=6, neuron=10, steps=50, lr=0.1, size=100, upscaling_steps=8, upscaling_factor=1.2)
    # view_deconvnet(cnn, img, num=5)
    # view_network_inversion(cnn, img)
    view_grad_cam(cnn, img, raw_img)


if __name__ == "__main__":
    main()

