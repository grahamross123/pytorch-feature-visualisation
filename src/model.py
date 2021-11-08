import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 16, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(16, 32, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(256, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def forward(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)

    def conv_fwd(self, x, layer="conv1", feature_idx=0):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x[0, feature_idx, :, :]
        return x.sum()


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.deConv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.deConv4 = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1, bias=False)
        self.deConv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1, bias=False)
        self.deConv2 = nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2, bias=False)
        self.deConv1 = nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2, bias=False)
        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2)
    
        self.feature_outputs = dict()
        self.switch_indices = dict()
        self.layer_names = []
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                self.feature_outputs[i] = x
                self.switch_indices[i] = indices
                self.layer_names.append(layer)
            else:
                x = layer(x)
                self.feature_outputs[i] = x
                self.layer_names.append(layer)
            if i == 11:
                h = x.register_hook(self.activations_hook)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_deconv(self, x, conv_layer):
        if conv_layer > 4 or conv_layer < 0:
            raise Exception(f"conv_layer should be between 1 and 12, not {conv_layer}")
        if conv_layer > 3:
            x = F.relu(self.deConv5(x))
        if conv_layer > 2:
            x = F.relu(self.deConv4(x))
        if conv_layer > 1:
            x = F.relu(self.deConv3(x))
            x = self.unpool(x, self.switch_indices[5], output_size=self.feature_outputs[4].shape[-2:])
        if conv_layer > 0:
            
            x = F.relu(self.deConv2(x))
            x = self.unpool(x, self.switch_indices[2], output_size=self.feature_outputs[1].shape[-2:])
        x = self.deConv1(x)
        # x = F.interpolate(x, size=(224, 224), mode="bilinear")
        return F.relu(x)


if __name__ == "__main__":   
    cnn = AlexNet()

    x = torch.randn(3, 224, 224).view(1, 3, 224, 224)
    output = cnn.forward(x)

    x = torch.rand(1, 256, 13, 13). view(1, 256, 13, 13)

    x = cnn.forward_deconv(x, 5)

    print(x.shape)

    # print(output.shape)