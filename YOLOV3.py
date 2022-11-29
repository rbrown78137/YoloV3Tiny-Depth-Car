import torch
import torch.nn as nn

import constants
# (Out channels, kernal_size, stride)
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    #Darknet - 53 before this
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=not bn_act,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1) #(0.1)
        self.use_bn_act = bn_act

    def forward(self,x):
        if self.use_bn_act:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, number_of_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(number_of_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels,channels//2,kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual =use_residual
        self.number_of_repeats = number_of_repeats

    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
               x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self,in_channels,number_of_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels,2*in_channels,kernel_size=3,padding=1),
            CNNBlock(2*in_channels,(number_of_classes+5) * 3,bn_act=False,kernel_size=1),
        )
        self.number_of_classes = number_of_classes

    def forward(self,x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, (self.number_of_classes + 5), x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOV3(nn.Module):
    def __init__(self, in_channels=1, number_of_classes=constants.number_of_classes):
        super().__init__()
        self.in_channels = in_channels
        self.number_of_classes = number_of_classes
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer,ResidualBlock) and layer.number_of_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer,nn.Upsample):
                x = torch.cat([x, route_connections[-1]],dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module,tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=1 if kernel_size == 3 else 0)
                )
                in_channels = out_channels
            elif isinstance(module,list):
                number_of_repeats = module[1]
                layers.append(ResidualBlock(in_channels, number_of_repeats=number_of_repeats))

            elif isinstance(module,str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, number_of_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, number_of_classes=self.number_of_classes)
                    ]
                    in_channels = in_channels//2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers

# if __name__ == "__main__":
#     num_classes = 2
#     IMAGE_SIZE = 416
#     model = YOLO(number_of_classes=num_classes)
#     x = torch.randn((2, 1, IMAGE_SIZE, IMAGE_SIZE))
#     out = model(x)
#     assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
#     assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
#     assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
#     print("Success!")