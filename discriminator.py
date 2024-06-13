import torch
import torch.nn as nn
import argparse

class CNNBlock(nn.Module):
    """
    A block that will be used, takes in input :
        * in_channels
        * out_channels
        * stride
    """
    def __init__(self,in_channels,out_channels,stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,padding=1,bias=False,padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    """
    * In the paper they specifiy that the first convlayer is without Batchnorm
    """
    def __init__(self,in_channels_x=3,in_channels_y=3, features = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels_x+in_channels_y,features[0],kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]: #Already use the first in the initial block
            layers.append(
                CNNBlock(in_channels,feature,stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"),
        )

        self.model = nn.Sequential(*layers)

    def forward(self,x,y):
        x = torch.cat([x,y],dim=1) #concatenate x and y along the first dim (that's why we use feautres*2)
        x = self.initial(x) #pass through first layer
        return self.model(x) #pass through the model
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size",type=int)
    parser.add_argument("--channels_1",type=int)
    parser.add_argument("--channels_2",type=int)
  
    args = parser.parse_args()
    H = args.size
    C1 = args.channels_1
    C2 = args.channels_2

    print(H,C1,C2)

    x = torch.rand((1,C1,H,H))
    y = torch.rand((1,C2,H,H))
    model = Discriminator(C1,C2)
    pred = model(x,y)
    print(pred.shape) #torch.Size([1, 512, 27, 27])




