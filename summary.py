# This part of the code is only used to view the network structure and is not test code.

from nets.unet import Unet
from utils.utils import net_flops


if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 21
    backbone        = 'resnet50'
    
    model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)

    # view network structure
    model.summary()

    # calculate network FLOPs
    net_flops(model, table=False)
    
    # get name and index of each layer
    for i,layer in enumerate(model.layers):
        print(i,layer.name)