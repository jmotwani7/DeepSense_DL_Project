from torchvision import models, transforms
import torch

class ResNet50Base:

    def __init__(self):
        model = models.resnet50(pretrained=True)
        print(model)
        print("----------------------")
        newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        print(newmodel)



def main():
    res = ResNet50Base()

if __name__ == "__main__":
    main()