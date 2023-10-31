import datetime
import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss, get_lr_scheduler)
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import UnetDataset
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import Iou_score, f_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Before training, carefully check whether your format meets the requirements. This library requires the dataset to be in VOC format, and you need to prepare both input images and labels.
1. Input images should be in .jpg format. There's no fixed size requirement; images will be automatically resized before training.
2. Grayscale images will be automatically converted to RGB for training, so there's no need for manual adjustments.
3. If the input image extensions are not .jpg, you need to batch convert them to .jpg before starting training.
4. Labels should be in .png format. There's no fixed size requirement; they will be automatically resized before training.
5. It needs to be changed so that the pixel value for the background is 0, and the pixel value for the target is 1.
'''

if __name__ == "__main__":    

    # eager mode option
    eager = False

    # GPU(s) used for training.
    # By default, it uses the first GPU. For two GPUs, use [0, 1]. For three GPUs, use [0, 1, 2].
    # When using multiple GPUs, the batch size on each GPU is the total batch size divided by the number of GPUs.
    train_gpu = [0,]

    # must be modified when training on your own dataset
    # number of classes + 1, e.g., 2 + 1
    num_classes = 2

    # backbone options: vgg, resnet50
    backbone = "resnet50"

    # model path
    model_path = "model_data/unet_resnet_voc.h5"

    # size of the input image, a multiple of 32
    input_shape = [512, 512]

    # Parameters for training during the freezing phase.
    # At this time, the backbone of the model is frozen, so the feature extraction network remains unchanged.
    # The memory occupied is relatively small, and only fine-tuning of the network is performed.
    # Init_Epoch:        The current starting training epoch of the model. Its value can be greater than Freeze_Epoch. For example, if set to:
    #                    Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100,
    #                    it will skip the freezing phase and start directly from epoch 60, adjusting the corresponding learning rate.
    #                    (Used for resuming training from a checkpoint)
    # Freeze_Epoch:      The epoch at which the model is trained in the frozen state.
    #                    (Becomes ineffective when Freeze_Train=False)
    # Freeze_batch_size: The batch size for training the model in the frozen state.
    #                    (Becomes ineffective when Freeze_Train=False)
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 5

    # Parameters for training during the unfreezing phase.
    # At this time, the backbone of the model is no longer frozen, so the feature extraction network will change.
    # The memory occupied is relatively large, and all parameters of the network will change.
    # UnFreeze_Epoch:      The total epochs for which the model will be trained.
    # Unfreeze_batch_size：The batch size for the model after unfreezing.
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 5
 
    # backbone is first frozen for training and then unfrozen for further training by default
    Freeze_Train = True

    # maximum learning rate: Adam = 1e-4, SGD = 1e-2
    Init_lr = 1e-4

    # minimum learning rate
    Min_lr = Init_lr * 0.01

    # optimizer type: adam, sgd
    optimizer_type = "adam"
    momentum = 0.9

    # learning rate decay type: step, cos
    lr_decay_type = 'cos'

    # save weights every 10 epochs
    save_period = 10
 
    # folder for saving logs and weights
    save_dir = 'logs'
    
    # dataset path
    VOCdevkit_path = 'VOCdevkit'

    # few classes, set to True
    # many classes and batch_size > 10, then set to True
    # many classes and batch_size < 10, then set to False
    dice_loss = True

    # focal loss option, to prevent imbalance between positive and negative samples
    focal_loss = False

    # assign different loss weights to different classes, balance by default
    cls_weights = np.ones([num_classes], np.float32)

    # Used to set whether to use multi-threading for data reading, 1 means turning off multi-threading.
    # Turning it on will speed up data reading but will consume more memory.
    # Turn on multi-threading when IO is the bottleneck, i.e., when the GPU computation speed is much faster than the speed of reading images.
    num_workers = 2

    # set GPU used
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    #  determine the number of GPUs currently in use compared to the actual number of GPUs on the machine
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")
        
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    
    if ngpus_per_node > 1:
        with strategy.scope():
            # get model
            model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
            if model_path != '':                
                # load weights
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        # get model
        model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
        if model_path != '':
            # load weights
            model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # read corresponding dataset txt
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    # loss function
    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    # The main feature extraction network has general features, and freeze training can speed up the training process.
    # It can also prevent the weights from being destroyed in the early stages of training.
    # Init_Epoch:     starting epoch
    # Freeze_Epoch:   epoch where training is froze
    # UnFreeze_Epoch: total training epoch
    # If you encounter OOM (Out of Memory) or insufficient GPU memory, please reduce the Batch_size.
    if True:
        if Freeze_Train:

            # freeze certain parts for training
            if backbone == "vgg":
                freeze_layers = 17
            elif backbone == "resnet50":
                freeze_layers = 172
            else:
                raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
            for i in range(freeze_layers): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

        # set batch_size to Unfreeze_batch_size if not performing freeze training
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        
        # determine current batch_size and adjust learning rate
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # get learning rate function
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small to continue training. Please expand the dataset.')

        train_dataloader = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)
        val_dataloader = UnetDataset(val_lines, input_shape, batch_size, num_classes, False, VOCdevkit_path)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        if eager:
            start_epoch = Init_Epoch
            end_epoch = UnFreeze_Epoch
            UnFreeze_flag = False

            gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
            if ngpus_per_node > 1:
                gen = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history = LossHistory(log_dir)

            # start training
            for epoch in range(start_epoch, end_epoch):

                # unfreeze the model and set parameters if it has frozen learning section
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_siz = Unfreeze_batch_size

                    # determine current batch_size and adjust learning rate
                    nbs = 16
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                    # get learning rate function
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model.layers)): 
                        model.layers[i].trainable = True

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                    train_dataloader.batch_size = batch_size
                    val_dataloader.batch_size = batch_size

                    gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    if ngpus_per_node > 1:
                        gen = strategy.experimental_distribute_dataset(gen)
                        gen_val = strategy.experimental_distribute_dataset(gen_val)
                    
                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)
                
                fit_one_epoch(model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, f_score(), save_period, save_dir, strategy)

                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()

        else:
            start_epoch = Init_Epoch
            end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
                
            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(loss = loss,
                            optimizer = optimizer,
                            metrics = [f_score()])
            else:
                model.compile(loss = loss,
                        optimizer = optimizer,
                        metrics = [f_score()])
                
            # Setting training parameters
            # logging:        set the save location for tensorboard
            # checkpoint:     set the details of weight saving, period is used to determine how many epochs to save once
            # lr_scheduler:   set the method of reducing the learning rate
            # early_stopping: set early stopping. If val_loss doesn't decrease for several times, training will automatically stop, indicating the model has basically converged
            time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(save_dir, "loss_" + str(time_str))
            logging = TensorBoard(log_dir)
            loss_history = LossHistory(log_dir)
            checkpoint = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x = train_dataloader,
                    steps_per_epoch = epoch_step,
                    validation_data = val_dataloader,
                    validation_steps = epoch_step_val,
                    epochs = end_epoch,
                    initial_epoch = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers = num_workers,
                    callbacks = callbacks
                )

            # unfreeze the model and set parameters if it has frozen learning section
            if Freeze_Train:
                batch_size = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch = UnFreeze_Epoch
                    
                # determine current batch_size and adjust learning rate
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                # get learning rate function
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]
                    
                for i in range(len(model.layers)): 
                    model.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(loss = loss,
                                optimizer = optimizer,
                                metrics = [f_score()])
                else:
                    model.compile(loss = loss,
                            optimizer = optimizer,
                            metrics = [f_score()])

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                train_dataloader.batch_size = Unfreeze_batch_size
                val_dataloader.batch_size = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x = train_dataloader,
                    steps_per_epoch = epoch_step,
                    validation_data = val_dataloader,
                    validation_steps = epoch_step_val,
                    epochs = end_epoch,
                    initial_epoch = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers = num_workers,
                    callbacks = callbacks
                )