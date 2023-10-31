import numpy as np
from PIL import Image

# Convert the image to RGB to prevent errors during prediction with grayscale images.
# The code only supports prediction with RGB images; all other types of images will be converted to RGB.
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#  resize input image
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
def preprocess_input(image):
    image = image / 127.5 - 1
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

# from https://github.com/ckyrkou/Keras_FLOP_Estimator 
# fix lots of bugs
def net_flops(model, table=False, print_result=True):
    if (table == True):
        print("\n")
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('=' * 120)
        
    # total FLOPs
    t_flops = 0
    factor = 1e9

    for l in model.layers:
        try:
            
            # initialize variables
            o_shape, i_shape, strides, ks, filters = ('', '', ''), ('', '', ''), (1, 1), (0, 0), 0
            flops = 0
            
            # get layer name            
            name = l.name
            
            if ('InputLayer' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            # reshape layer
            elif ('Reshape' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # padding layer
            elif ('Padding' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # flatten layer
            elif ('Flatten' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            # activation layer
            elif 'Activation' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            # LeakyReLU
            elif 'LeakyReLU' in str(l):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops += i_shape[0] * i_shape[1] * i_shape[2]
                    
            # max pooling layer
            elif 'MaxPooling' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                    
            # average pooling layer
            elif ('AveragePooling' in str(l) and 'Global' not in str(l)):
                strides = l.strides
                ks = l.pool_size
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops += o_shape[0] * o_shape[1] * o_shape[2]

            # global average pooling layer
            elif ('AveragePooling' in str(l) and 'Global' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops += (i_shape[0] * i_shape[1] + 1) * i_shape[2]
                
            # batch normalization
            elif ('BatchNormalization' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    temp_flops = 1
                    for i in range(len(i_shape)):
                        temp_flops *= i_shape[i]
                    temp_flops *= 2
                    
                    flops += temp_flops
                
            # fully connected layer
            elif ('Dense' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                
                    temp_flops = 1
                    for i in range(len(o_shape)):
                        temp_flops *= o_shape[i]
                        
                    if (i_shape[-1] == None):
                        temp_flops = temp_flops * o_shape[-1]
                    else:
                        temp_flops = temp_flops * i_shape[-1]
                    flops += temp_flops

            # simple convolutional layer
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                bias = 1 if l.use_bias else 0
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] * i_shape[2] + bias)

            # depthwise convolutional layer
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                bias = 1 if l.use_bias else 0
            
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias)
                
            # depthwise separable convolutional layer
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += i_shape[2] * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias) + \
                             filters * o_shape[0] * o_shape[1] * (1 * 1 * i_shape[2] + bias)

            # when model exists in model
            elif 'Model' in str(l):
                flops = net_flops(l, print_result=False)
                
            t_flops += flops

            if (table == True):
                print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                    name[:25], str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
                
        except:
            pass
    
    t_flops = t_flops * 2
    if print_result:
        show_flops = t_flops / factor
        print('Total GFLOPs: %.3fG' % (show_flops))
    return t_flops