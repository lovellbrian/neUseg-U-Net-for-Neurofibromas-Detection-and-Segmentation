import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


# If you want to increase the test set, modify trainval_percent 
# Modify train_percent to change the ratio of the validation set, e.g., 9:1
# Currently, this library uses the test set as the validation set, without a separate test set division.
trainval_percent = 1
train_percent = 0.9

# direct to the folder where the VOC dataset is located.
# direct to the VOC dataset in the root directory by default
VOCdevkit_path = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)  
    list = range(num)  
    tv = int(num*trainval_percent)  
    tr = int(tv*train_percent)  
    trainval = random.sample(list,tv)  
    train = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    classes_nums = np.zeros([256], np.int64)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not detected, please check if the file exists in the specific path and if the suffix is png."%(png_file_name))
        
        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The label image %s has a shape of %s, which is not a grayscale or 8-bit color image. Please carefully check the dataset format."%(name, str(np.shape(png))))
            print("The label image needs to be a grayscale or 8-bit color image, and the value of each pixel in the label is the category to which this pixel belongs."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print pixel values and counts.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected that the pixel values in the label only contain 0 and 255, the data format is incorrect.")
        print("For binary classification problems, the label needs to be modified so that the pixel value of the background is 0, and the pixel value of the target is 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Detected that the label only contains background pixels, the data format is incorrect. Please carefully check the dataset format.")