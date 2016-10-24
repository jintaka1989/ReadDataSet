# coding: UTF-8
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

NUM_CLASSES = 4
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('labels', 'labels.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

class ImageDataSet:
    def  __init__(self, filename):
        # データを入れる配列
        self.train_image = []
        self.train_label = []
        self.image = []

        self.f = open(filename, 'r')
        num_lines = sum(1 for one_line in self.f)
        self.f.close()

        self.f = open(filename, 'r')
        for line in self.f:
            line = line.replace("\n", " ")
            l = line.split()
            img = cv2.imread(l[0])
            img = cv2.resize(img, (28, 28))
            self.image.append(img.astype(np.float32)/255)
            self.train_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(num_lines)
            tmp[int(l[1])] = 1
            self.train_label.append(tmp)
        self.train_image = np.asarray(self.train_image)
        self.train_label = np.asarray(self.train_label)
        self.f.close()
