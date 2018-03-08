'''
wrap deeplab-v3 model
'''

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np
import cv2

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'



def main():
    pass



def video_seg():
    pass



def infer_img(sess,img):
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    img = tf.placeholder([None,None,None])

    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    


def load_model(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()    
    sess.run(init)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))



def net_init(model_path = ''):
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)
    #net = DeepLabResNetModel({'data': tf.placeholder([None,None,None,None],dtype=tf.float32) }, 
    #                            is_training=False, 
    #                            num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load_model(loader, sess, model_path)
    
    # Perform inference.
    preds = sess.run(pred)
    
    msk = decode_labels(preds, num_classes=args.num_classes)
    return msk





if __name__=="__main__":
    main()
    
