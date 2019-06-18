from scipy import misc
import numpy as np
import cv2
import os
from tensorflow.python.keras.layers import Input
from model import VGG16
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def rgba2rgb(img):
    return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def padding(x):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    return temp_x

# [imread](https://blog.csdn.net/renelian1572/article/details/78761278)
def load_image(path):
    x = misc.imread(path)
    if x.shape[2] == 4:
       x = rgba2rgb(x)
    sh = x.shape
    # Zero-center by mean pixel
    g_mean = np.array(([103.939,116.779,123.68])).reshape([1,1,3])
    x = padding(x)
    x = misc.imresize(x.astype(np.uint8), target_size, interp="bilinear").astype(np.float32) - g_mean
    x = np.expand_dims(x,0)
    return x,sh

def cut(pridict,shape):
    h,w,c = shape
    size = max(h, w)
    pridict = cv2.resize(pridict, (size,size))
    paddingh = (size - h) // 2
    paddingw = (size - w) // 2
    return pridict[paddingh:h + paddingh, paddingw:w + paddingw]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getres(pridict,shape):
    pridict = sigmoid(pridict)*255
    pridict = np.array(pridict, dtype=np.uint8)
    pridict = np.squeeze(pridict)
    pridict = cut(pridict, shape)
    return pridict

def laplace_edge(x):
    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(x/255.,-1,laplace)
    edge = np.maximum(np.tanh(edge),0)
    edge = edge * 255
    edge = np.array(edge, dtype=np.uint8)
    return edge

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'model/PFA_00050.h5'

target_size = (256,256)

dropout = False
with_CPFE = True
with_CA = True
with_SA = True

if target_size[0 ] % 32 != 0 or target_size[1] % 32 != 0:
    raise ValueError('Image height and wight must be a multiple of 32')

model_input = Input(shape=(target_size[0],target_size[1],3))
model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, with_CA=with_CA, with_SA=with_SA)
model.load_weights(model_name,by_name=True)

for layer in model.layers:
    layer.trainable = False
'''
image_path = 'image/2.jpg'
img, shape = load_image(image_path)
img = np.array(img, dtype=np.float32)
sa = model.predict(img)
sa = getres(sa, shape)
plt.title('saliency')
plt.subplot(131)
plt.imshow(cv2.imread(image_path))
plt.subplot(132)
plt.imshow(sa,cmap='gray')
plt.subplot(133)
edge = laplace_edge(sa)
plt.imshow(edge,cmap='gray')
plt.savefig(os.path.join('./train_1000_output','alpha.png'))
#misc.imsave(os.path.join('./train_1000_output','alpha.png'), sa)
'''
HOME = os.path.expanduser('~')
rgb_folder = os.path.join(HOME, 'data/train_1000/image_1000')
output_folder = './train_1000_output'
rgb_names = os.listdir(rgb_folder)
print(rgb_folder, "\nhas {0} pics.".format(len(rgb_names)))
for rgb_name in rgb_names:    
    if rgb_name[-4:] == '.jpg':
        img, shape = load_image(os.path.join(rgb_folder, rgb_name))
        img = np.array(img, dtype=np.float32)
        sa = model.predict(img)
        sa = getres(sa, shape)
        misc.imsave(os.path.join(output_folder, rgb_name), sa)
