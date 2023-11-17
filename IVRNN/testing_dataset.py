#
# Copyright (c) Facebook, Inc. and its affiliates.
#
from glob import glob
import os
import time
import random
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import kl_divergence, Normal, MultivariateNormal
from tqdm import tqdm

import train_fns
import utils
import configs
from utils import torch2img, save_gif
from logger import MyLogger
from datasets import get_dataset
from init_model import init_model

#config['seq_len'] = config['n_ctx'] + config['n_steps']

'''
normalize = True
torch.backends.cudnn.benchmark = True

from dataloaders.pushbair_loader import PushDataset

train_dataset = PushDataset(
    'train',
    8,#config['seq_len'],
    normalize=normalize,
)

val_dataset = PushDataset(
    'test',
    8,#config['seq_len'],
    normalize=normalize,
)

img_ch = 3
'''
data_dir = os.path.join('processed_video', 'train')
print(data_dir)
example_dirs = glob(os.path.join(data_dir, '*', '*'))
print(len(example_dirs))
#data_dir = '%s/softmotion30_44k/%s' % ('./BAIR', 'test')
#print(data_dir)
#for i in range(30):
#    print(i)
'''
import os
import io

import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

#from scipy.misc import imresize
#from scipy.misc import imsave

import imageio
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='base directory to save processed data')
opt = parser.parse_args()

def get_seq(dname):
    data_dir = '%s/softmotion30_44k/%s' % (opt.data_dir, dname)

    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k=0
        for serialized_example in tf.compat.v1.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                #img = Image.open(io.BytesIO(byte_str))
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                image_seq.append(arr.reshape(1, 64, 64, 3)/255.)
            image_seq = np.concatenate(image_seq, axis=0)
            k=k+1
            yield f, k, image_seq
seq_generator = get_seq('test')
f, k, seq = next(seq_generator)
f = f.split('/')[-1]
print(f[:-10])
'''
