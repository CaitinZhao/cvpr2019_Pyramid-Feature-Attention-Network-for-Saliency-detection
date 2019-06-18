import numpy as np
import cv2
import random
import os

def padding(x,y):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_y = np.zeros((size,size))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    temp_y[paddingh:h+paddingh,paddingw:w+paddingw] = y
    return temp_x,temp_y

def random_crop(x,y):
    h,w = y.shape
    randh = np.random.randint(h/8)
    randw = np.random.randint(w/8)
    randf = np.random.randint(10)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
    if randf >= 5:
        x = x[::, ::-1, ::]
        y = y[::, ::-1]
    return x[p0:p1,p2:p3],y[p0:p1,p2:p3]

def random_rotate(x,y):
    angle = np.random.randint(-25,25)
    h, w = y.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h))

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def getTrainGenerator(file_path, target_size, batch_size, israndom=False):
    f = open(file_path, 'r')
    trainlist = f.readlines()
    f.close()
    while True:
        random.shuffle(trainlist)
        batch_x = []
        batch_y = []
        for name in trainlist:
            p = name.strip('\r\n').split(',')
            img_path = p[0]
            mask_path = p[1]
            x = cv2.imread(img_path)
            y = cv2.imread(mask_path)
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if len(y.shape) == 3:
                y = y[:,:,0]
            y = y/y.max()
            if israndom:
                x,y = random_crop(x,y)
                x,y = random_rotate(x,y)
                x = random_light(x)

            x = x[..., ::-1]
            # Zero-center by mean pixel
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            x, y = padding(x, y)

            x = cv2.resize(x, target_size, interpolation=cv2.INTER_LINEAR)
            y = cv2.resize(y, target_size, interpolation=cv2.INTER_NEAREST)
            y = y.reshape((target_size[0],target_size[1],1))
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == batch_size:
                yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))
                batch_x = []
                batch_y = []

def ge_train_pair(filename, train_dir, image_dir, mask_dir):
    f = open(filename,'w')
    path = os.path.join(train_dir, image_dir)
    name = os.listdir(path)
    print(path, "\nhas {0} images.".format(len(name)))
    for n in name:
        if n[-4:] == '.jpg':
            f.writelines([os.path.join(train_dir, image_dir, str(n)), ',',
                             os.path.join(train_dir, mask_dir, str(n[:-4] + '.png')), '\n'])
        else:
            pass
    f.close()


