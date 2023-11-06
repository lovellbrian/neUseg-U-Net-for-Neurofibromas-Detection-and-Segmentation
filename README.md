# neUseg: U-Net for Neurofibromas Detection and Segmentation

This repository stores the code of my project for REIT4841.

| Folder/Script | Description |
| :------------ | :-----------|
| VOCdevkit/VOC2007 | Include “ImageSets/Segmentation”, “JPEGImages”, and “SegmentationClass”. Refer to Section 4.3.1 for their usage |
| logs | Store model weights and loss data. | 
| nets | Include code of U-Net, VGG16, and ResNet-50. | 
| utils | Includes utility codes. |
| get_miou.py | Calculate evaluation metrics and generate corresponding bar charts. |
| gui.py | Graphical user interface. |
| gui.ui | Graphical user interface. |
| summary.py | Display network structure. |
| train.py | Training code for the model. |
| unet.py | Predict segmentation and get mIoU metrics. |
| voc_annotation.py | Set train-validation ratio and generate corresponding .txt files of their filenames. |
