import colorsys
import copy
import time

import cv2
import numpy as np
from PIL import Image

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# modify model_path and num_classes for custom trained model
# if there's a shape mismatch, make sure to adjust the model_path and num_classes during training
class Unet(object):
    _defaults = {
        # model_path directs to the weight file in the logs folder
        # after training, there are multiple weight files in the logs folder, choose the one with lower validation loss
        # lower validation loss doesn't mean higher miou, it just means that the weights generalize better on the validation set
        'model_path':'model_data/NF1.h5',

        # number of classes + 1
        "num_classes" : 2,

        # backbone options: vgg, resnet50
        "backbone" : "resnet50",

        # input image size
        "input_shape" : [512, 512],

        # mix_type parameter is used to control the visualization method of the detection result.
        # mix_type = 0 means mixing the original image with the generated image
        # mix_type = 1 means only keeping the generated image
        # mix_type = 2 means only removing the background and keeping the target in the original image
        "mix_type" : 0,
    }

    # initialize U-Net
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # set different colors for bounding boxes
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # get model
        self.generate()

        show_config(**self._defaults)

    # load model
    def generate(self):

        # load model and weights
        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)

        self.model.load_weights(self.model_path)

    # detect image
    def detect_image(self, image, model_path='model_data/NF1.h5',count=False, name_classes=None,):

        # convert image to RGB here to prevent errors during prediction with grayscale images
        # code only supports RGB image prediction, all other types of images will be converted to RGB
        self.model.load_weights(model_path)
        print('{} model loaded.'.format(model_path))
        image = cvtColor(image)
        
        # backup input image for drawing later
        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]
        
        # add gray bars to image to achieve non-distorted resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        
        # normalize and add batch_size dimension
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        # pass image through the network for prediction
        pr = self.model.predict(image_data)[0]

        # crop out gray bar part
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        
        # resize image
        pr = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        
        # get pixel category
        pr = pr.argmax(axis=-1)
        
        # count
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])

            # convert new image to Image format
            image = Image.fromarray(np.uint8(seg_img))

            # blend new image with original image
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])

            # convert new image to Image format
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            
            # convert new image to Image format
            image = Image.fromarray(np.uint8(seg_img))

        return image,classes_nums

    def get_FPS(self, image, test_interval):

        # convert image to RGB here to prevent errors during prediction with grayscale images
        # code only supports RGB image prediction, all other types of images will be converted to RGB
        image = cvtColor(image)

        # add gray bars to image to achieve non-distorted resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # normalize and add batch_size dimension
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        # pass image through the network for prediction
        pr = self.model.predict(image_data)[0]
        
        # crop out gray bar part
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        # get pixel category
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
                
        t1 = time.time()
        for _ in range(test_interval):
            
            # pass image through the network for prediction
            pr = self.model.predict(image_data)[0]

            # crop out gray bar part
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

            # get pixel category
            pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
    def get_miou_png(self, image):
        
        # convert image to RGB here to prevent errors during prediction with grayscale images
        # code only supports RGB image prediction, all other types of images will be converted to RGB
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]
        
        # add gray bars to image to achieve non-distorted resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        
        # normalize and add batch_size dimension
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        # pass image through the network for prediction
        pr = self.model.predict(image_data)[0]

        # crop out gray bar part
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        
        # resize image        
        pr = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        
        # get pixel category
        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image