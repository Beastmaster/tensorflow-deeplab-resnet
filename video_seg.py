'''
wrap deeplab-v3 model
'''

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
import cv2
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

from matplotlib import pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'



def main():
    #test_img()
    video_seg()


def video_seg():
    model = net_init(model_path='Model/deeplab_resnet.ckpt')

    video_file = 'E:/runable/all_video_data/videodata2/1_xsf_0.mp4'
    video_file = 'E:/runable/all_video_data/xusongfei_fall/VID_20171201_223352.mp4'
    cap = cv2.VideoCapture(video_file)
    ret,frame = cap.read()

    bname = os.path.basename(video_file).split('.')[0]
    outfile = os.path.join('./images',bname+'.avi')
    ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(outfile,fourcc, 20, (ww,hh))

    while frame is not None:
        label = infer_img(model,frame)
        frame[label>50,:] = [255,255,0]
        out.write(frame)
        cv2.imshow("image",frame)
        if cv2.waitKey(1) == 27:
            break
        ret,frame = cap.read()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        



def test_img(fname = ''):
    model = net_init(model_path='Model/deeplab_resnet.ckpt')
    fname='images/test.png'
    img = cv2.imread(fname)
    nimg = infer_img(model,img)
    
    cv2.imshow("image",nimg)
    cv2.waitKey(0)


def bgr_demean(img):
    '''
    3 dimensions
    '''
    nimage = np.zeros_like(img,dtype=np.float32)
    img.astype(np.float32)
    img = img[..., [2,1,0]]
    return img



def infer_img(model,img):
    sess = model[0]
    feeds= model[1]
    img = bgr_demean(img)
    shape = (img.shape[1],img.shape[0])
    img = img[np.newaxis,...]
    # Perform inference.
    preds = sess.run(feeds['pred'], feed_dict = {feeds['data']: img})
    # gray scale conversion
    preds = np.squeeze(preds)==15
    preds=np.uint8(preds)*255
    nimg = cv2.resize(preds,shape,interpolation=cv2.INTER_LINEAR)
    nimg[nimg>127] = 255
    return nimg
    
    


def load_model(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    # Set up TF session and initialize variables. 
    init = tf.global_variables_initializer()    
    sess.run(init)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))



def net_init(model_path = ''):
    img = tf.placeholder(dtype=tf.float32, shape=[None,None,None,3])
    # Create network.
    net = DeepLabResNetModel({'data': img}, is_training=False, num_classes=NUM_CLASSES)
    #net = DeepLabResNetModel( img, 
    #                            is_training=False, 
    #                            num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.argmax(raw_output, dimension=-1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load_model(loader, sess, model_path)
    
    return sess,{'data':img, 'pred': raw_output_up}
    






if __name__=="__main__":
    main()
    
