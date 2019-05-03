from keras import callbacks, optimizers
import tensorflow as tf
import os
from keras.layers import Input
from model import VGG16
from data import getTrainGenerator
from utils import *
from edge_hold_loss import *
import math

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def lr_scheduler(epoch):
    drop = 0.5
    epoch_drop = epochs/8.
    lr = base_lr * math.pow(drop, math.floor((1+epoch)/epoch_drop))
    print('lr: %f' % lr)
    return lr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model your dataset')
    parser.add_argument('--train_file',default='train_pair.txt',help='your train file', type=str)
    parser.add_argument('--model_weights',default='model/vgg16_no_top.h5',help='your model weights', type=str)

    args = parser.parse_args()
    model_name = args.train_file
    '''
    the from of 'train_pair.txt' is 
    img_path1 gt_path1\n
    img_path2 gt_path2\n 
    '''
    train_path = args.model_weights
    
    print "train_file", train_path
    print "model_weights", model_name
    
    target_size = (256,256)
    batch_size = 15
    base_lr = 1e-2
    epochs = 50

    f = open(train_path, 'r')
    trainlist = f.readlines()
    f.close()
    steps_per_epoch = len(trainlist)/batch_size

    optimizer = optimizers.SGD(lr=base_lr, momentum=0.9, decay=0)
    # optimizer = optimizers.Adam(lr=base_lr)
    loss = EdgeHoldLoss

    metrics = [acc,pre,rec]
    dropout = True
    with_CPFE = True
    with_CA = True
    with_SA = True
    log = './PFA.csv'
    tb_log = './tensorboard-logs/PFA'
    model_save = 'model/PFA_'
    model_save_period = 5

    if target_size[0 ] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('Image height and wight must be a multiple of 32')

    traingen = getTrainGenerator(train_path, target_size, batch_size, israndom=True)

    model_input = Input(shape=(target_size[0],target_size[1],3))
    model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, with_CA=with_CA, with_SA=with_SA)
    for i,layer in enumerate(model.layers):
        print i,layer.name
    model.load_weights(model_name,by_name=True)

    tb = callbacks.TensorBoard(log_dir=tb_log)
    lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)
    es = callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')
    modelcheck = callbacks.ModelCheckpoint(model_save+'{epoch:05d}.h5', monitor='loss', verbose=1,
        save_best_only=False, save_weights_only=True, mode='auto', period=model_save_period)
    callbacks = [lr_decay,modelcheck,tb]

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    model.fit_generator(traingen, steps_per_epoch=steps_per_epoch,
                        epochs=epochs,verbose=1,callbacks=callbacks)
