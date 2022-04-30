from torch import nn
import torchvision
from model.upsampling import FastUpConvolution


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Resnet50BasedModel(nn.Module):
    """"
    Resnet 50 Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        Resnet Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, freeze_decoder_weights=False):
        super(Resnet50BasedModel, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.freeze_decoder_weights = freeze_decoder_weights

        if self.freeze_decoder_weights:
            for param in resnet50.parameters():
                param.requires_grad = False

        # remove until the resnet-50 AvgPool Layer
        self.model = nn.Sequential(*(list(resnet50.children())[:-2]),
                                   nn.Conv2d(2048, 1024, (1, 1)),
                                   nn.BatchNorm2d(1024), FastUpConvolution(1024, 512),
                                   FastUpConvolution(512, 256),
                                   FastUpConvolution(256, 128),
                                   FastUpConvolution(128, 64), nn
                                   .Conv2d(64, 1, (3, 3), padding='same'),
                                   nn.ReLU())
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor
