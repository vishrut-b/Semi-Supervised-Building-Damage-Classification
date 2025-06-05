#!/usr/bin/env python3

path = './Data/Unlabeled images'

from PIL import Image
import glob
import numpy as np

image_list = []
for filename in glob.glob(path + '/*.jpg'):
    im = Image.open(filename)
    im = im.resize((224,224))
    image_list.append(im)

images_np = np.array([np.array(im)[:,:,::-1] for im in image_list]) # transform to bgr to match training set

# Shuffle and shorten (for speed, for now)
np.random.seed(10)
indices = np.arange(0, len(images_np))
test_indices = np.random.randint(0,len(images_np),(200,))
train_indices = np.delete(indices, test_indices)
X_test_unlabeled = images_np.take(test_indices,axis=0)
X_train_unlabeled = images_np.take(train_indices,axis=0)


np.save('./Data/task0/task0_X_train.npy',X_train_unlabeled)
np.save('./Data/task0/task0_X_test.npy',X_test_unlabeled)

from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_test_unlabeled = np.load('./Data/task0/task0_X_test.npy')

task1labels= [];
task2labels= [];

for im_array in X_test_unlabeled:
    imshow(im_array[:,:,::-1]) # images in bgr
    plt.show()
    plt.close()
    
    
task1labels = [2,0,2,0,1,2,2,2,2,2,0,0,0,2,0,2,0,2,2,2,2,2,2,2,0,0,2,2,2,2,1,
               2,2,2,2,2,2,0,2,2,2,2,0,2,2,2,2,2,2,2,0,2,2,2,0,0,2,0,1,0,2,2,
               2,0,0,0,1,0,0,2,0,2,2,0,2,0,2,0,2,2,0,2,0,2,0,0,0,0,2,2,0,0,2,
               0,2,0,0,0,0,2,0,0,2,0,2,0,0,2,2,0,0,1,0,0,2,0,0,2,0,0,0,0,0,0,
               0,2,0,0,2,0,2,0,2,0,2,0,0,2,0,2,0,2,0,0,2,2,1,2,0,1,0,2,0,2,0,
               0,2,2,2,2,0,2,0,2,2,0,2,2,0,2,1,2,2,2,1,0,2,0,0,0,0,2,2,2,0,2,
               0,2,0,0,2,0,2,2,0,0,2,0,0,0]

task2labels = [1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,
               0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
               1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,
               0,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,
               0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,1,1,
               1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,
               1,1,0,0,1,1,0,0,0,0,1,0,1,0]

y_test_unlabeled_1 = to_categorical(np.array(task1labels))
y_test_unlabeled_2 = to_categorical(np.array(task2labels))

np.save('./Data/task0/y_test_unlabeled_1.npy',y_test_unlabeled_1)
np.save('./Data/task0/y_test_unlabeled_2.npy',y_test_unlabeled_2)
    
r = np.random.randint(0,len(image_list))
r_im = image_list[r]
r_im.show()
r_im_np = np.array(r_im)
r_im2 = Image.fromarray(r_im_np)
r_im2.show()


